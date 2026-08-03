// Copyright 2026 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures::{StreamExt, TryStreamExt};
use iceberg::io::{
    FileMetadata, FileRead, FileWrite, InputFile, OutputFile, Storage, StorageConfig,
    StorageFactory,
};
use iceberg::{Error as IcebergError, ErrorKind as IcebergErrorKind, Result as IcebergResult};
use object_store::azure::{AzureCredentialProvider, MicrosoftAzureBuilder};
use object_store::buffered::BufWriter;
use object_store::gcp::{GcpCredentialProvider, GoogleCloudStorageBuilder, GoogleConfigKey};
use object_store::path::Path;
use object_store::{ObjectStore as OSObjectStore, ObjectStoreExt};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use url::Url;

use crate::azure_sas_provider::AzureBrokerConfig;
use crate::cloud_credential_cache::{cached_azure_sas_provider, cached_gcp_credential_provider};
use crate::gcp_impersonation::REFRESH_OFFSET_SECS;

const GCS_SERVICE_PATH: &str = "gcs.service.path";
const ADLS_ENDPOINT_SUFFIX: &str = "adls.endpoint-suffix";
const DEFAULT_ADLS_ENDPOINT_SUFFIX: &str = "core.windows.net";

#[derive(Clone, Debug, Deserialize, Serialize)]
enum RefreshingCloudProvider {
    Azure {
        cache_key: Option<String>,
        broker_config: AzureBrokerConfig,
    },
    Gcp {
        cache_key: Option<String>,
        target_sa: String,
        token_lifetime_secs: u64,
    },
}

/// Iceberg's OpenDAL factory accepts a request-time AWS credential loader, but
/// its Azure and GCS variants only accept static tokens. This factory routes
/// those two impersonation modes through object_store instead, whose builders
/// retain a dynamic CredentialProvider and consult it for every request.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct RefreshingCloudStorageFactory {
    provider: RefreshingCloudProvider,
}

impl RefreshingCloudStorageFactory {
    pub(crate) fn azure(cache_key: Option<String>, broker_config: AzureBrokerConfig) -> Self {
        Self {
            provider: RefreshingCloudProvider::Azure {
                cache_key,
                broker_config,
            },
        }
    }

    pub(crate) fn gcp(
        cache_key: Option<String>,
        target_sa: String,
        token_lifetime_secs: u64,
    ) -> Self {
        Self {
            provider: RefreshingCloudProvider::Gcp {
                cache_key,
                target_sa,
                token_lifetime_secs,
            },
        }
    }
}

#[typetag::serde(name = "RefreshingCloudStorageFactory")]
impl StorageFactory for RefreshingCloudStorageFactory {
    fn build(&self, config: &StorageConfig) -> IcebergResult<Arc<dyn Storage>> {
        Ok(Arc::new(RefreshingCloudStorage {
            provider: self.provider.clone(),
            props: config.props().clone(),
        }))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RefreshingCloudStorage {
    provider: RefreshingCloudProvider,
    props: HashMap<String, String>,
}

struct StoreLocation {
    store: Arc<dyn OSObjectStore>,
    path: Path,
}

impl RefreshingCloudStorage {
    async fn store_for_path(&self, path: &str) -> IcebergResult<StoreLocation> {
        match &self.provider {
            RefreshingCloudProvider::Azure {
                cache_key,
                broker_config,
            } => {
                let endpoint_suffix = self
                    .props
                    .get(ADLS_ENDPOINT_SUFFIX)
                    .filter(|suffix| !suffix.is_empty())
                    .map(String::as_str)
                    .unwrap_or(DEFAULT_ADLS_ENDPOINT_SUFFIX);
                let location = parse_azure_location(
                    path,
                    broker_config.account_name(),
                    broker_config.bucket(),
                    endpoint_suffix,
                )?;
                let provider =
                    cached_azure_sas_provider(cache_key.as_deref(), broker_config.clone())
                        .await
                        .map_err(|error| {
                            unexpected(
                                "Azure SAS provider initialization failed",
                                std::io::Error::other(error.to_string()),
                            )
                        })?;
                let credentials: AzureCredentialProvider = provider;
                let store = MicrosoftAzureBuilder::new()
                    .with_account(location.account)
                    .with_container_name(location.container)
                    .with_endpoint(location.endpoint)
                    .with_allow_http(false)
                    .with_credentials(credentials)
                    .build()
                    .map_err(|error| unexpected("Azure object store construction failed", error))?;
                Ok(StoreLocation {
                    store: Arc::new(store),
                    path: location.path,
                })
            }
            RefreshingCloudProvider::Gcp {
                cache_key,
                target_sa,
                token_lifetime_secs,
            } => {
                let (bucket, object_path) = parse_gcp_location(path)?;
                let provider = cached_gcp_credential_provider(
                    cache_key.as_deref(),
                    target_sa.clone(),
                    Duration::from_secs(*token_lifetime_secs),
                    Duration::from_secs(REFRESH_OFFSET_SECS),
                )
                .await
                .map_err(|error| {
                    unexpected(
                        "GCP credential provider initialization failed",
                        std::io::Error::other(error.to_string()),
                    )
                })?;
                let credentials: GcpCredentialProvider = provider;
                let mut builder = GoogleCloudStorageBuilder::new()
                    .with_bucket_name(bucket)
                    .with_credentials(credentials);
                if let Some(endpoint) = self.props.get(GCS_SERVICE_PATH) {
                    builder = builder.with_config(GoogleConfigKey::BaseUrl, endpoint.clone());
                }
                let store = builder
                    .build()
                    .map_err(|error| unexpected("GCS object store construction failed", error))?;
                Ok(StoreLocation {
                    store: Arc::new(store),
                    path: object_path,
                })
            }
        }
    }
}

#[typetag::serde(name = "RefreshingCloudStorage")]
#[async_trait::async_trait]
impl Storage for RefreshingCloudStorage {
    async fn exists(&self, path: &str) -> IcebergResult<bool> {
        let location = self.store_for_path(path).await?;
        match location.store.head(&location.path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(error) => Err(from_object_store_error(error)),
        }
    }

    async fn metadata(&self, path: &str) -> IcebergResult<FileMetadata> {
        let location = self.store_for_path(path).await?;
        let metadata = location
            .store
            .head(&location.path)
            .await
            .map_err(from_object_store_error)?;
        Ok(FileMetadata {
            size: metadata.size,
        })
    }

    async fn read(&self, path: &str) -> IcebergResult<bytes::Bytes> {
        let location = self.store_for_path(path).await?;
        location
            .store
            .get(&location.path)
            .await
            .map_err(from_object_store_error)?
            .bytes()
            .await
            .map_err(from_object_store_error)
    }

    async fn reader(&self, path: &str) -> IcebergResult<Box<dyn FileRead>> {
        let location = self.store_for_path(path).await?;
        Ok(Box::new(ObjectStoreFileRead { location }))
    }

    async fn write(&self, path: &str, bs: bytes::Bytes) -> IcebergResult<()> {
        let location = self.store_for_path(path).await?;
        location
            .store
            .put(&location.path, bs.into())
            .await
            .map_err(from_object_store_error)?;
        Ok(())
    }

    async fn writer(&self, path: &str) -> IcebergResult<Box<dyn FileWrite>> {
        let location = self.store_for_path(path).await?;
        Ok(Box::new(ObjectStoreFileWrite {
            writer: BufWriter::new(location.store, location.path),
        }))
    }

    async fn delete(&self, path: &str) -> IcebergResult<()> {
        let location = self.store_for_path(path).await?;
        location
            .store
            .delete(&location.path)
            .await
            .map_err(from_object_store_error)
    }

    async fn delete_prefix(&self, path: &str) -> IcebergResult<()> {
        let location = self.store_for_path(path).await?;
        let objects = location
            .store
            .list(Some(&location.path))
            .map_ok(|metadata| metadata.location);
        location
            .store
            .delete_stream(objects.boxed())
            .try_collect::<Vec<_>>()
            .await
            .map_err(from_object_store_error)?;
        Ok(())
    }

    fn new_input(&self, path: &str) -> IcebergResult<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> IcebergResult<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

struct ObjectStoreFileRead {
    location: StoreLocation,
}

#[async_trait::async_trait]
impl FileRead for ObjectStoreFileRead {
    async fn read(&self, range: std::ops::Range<u64>) -> IcebergResult<bytes::Bytes> {
        self.location
            .store
            .get_range(&self.location.path, range)
            .await
            .map_err(from_object_store_error)
    }
}

struct ObjectStoreFileWrite {
    writer: BufWriter,
}

#[async_trait::async_trait]
impl FileWrite for ObjectStoreFileWrite {
    async fn write(&mut self, bs: bytes::Bytes) -> IcebergResult<()> {
        self.writer.put(bs).await.map_err(from_object_store_error)
    }

    async fn close(&mut self) -> IcebergResult<()> {
        self.writer
            .shutdown()
            .await
            .map_err(|error| unexpected("object store writer close failed", error))
    }
}

struct AzureLocation {
    account: String,
    container: String,
    endpoint: String,
    path: Path,
}

fn parse_azure_location(
    absolute_path: &str,
    expected_account: &str,
    expected_container: &str,
    expected_suffix: &str,
) -> IcebergResult<AzureLocation> {
    let url = Url::parse(absolute_path).map_err(|error| {
        invalid(format!(
            "Invalid Azure storage URL {absolute_path}: {error}"
        ))
    })?;
    let expected_service = match url.scheme() {
        "abfss" => "dfs",
        "wasbs" => "blob",
        "abfs" | "wasb" => {
            return Err(invalid(format!(
                "Insecure Azure scheme is not allowed with broker credentials: {}",
                url.scheme()
            )));
        }
        scheme => return Err(invalid(format!("Unsupported Azure scheme: {scheme}"))),
    };
    let container = url.username();
    if container.is_empty() {
        return Err(invalid(format!(
            "Azure storage URL is missing a container: {absolute_path}"
        )));
    }
    if container != expected_container {
        return Err(invalid(format!(
            "Azure container mismatch: credential broker configured {expected_container}, path {container}"
        )));
    }
    let host = url.host_str().ok_or_else(|| {
        invalid(format!(
            "Azure storage URL is missing an account endpoint: {absolute_path}"
        ))
    })?;
    let mut host_parts = host.splitn(3, '.');
    let account = host_parts.next().unwrap_or_default();
    let service = host_parts.next().unwrap_or_default();
    let suffix = host_parts.next().unwrap_or_default();
    if account.is_empty() || service != expected_service || suffix.is_empty() {
        return Err(invalid(format!(
            "Invalid Azure storage endpoint in {absolute_path}"
        )));
    }
    if account != expected_account {
        return Err(invalid(format!(
            "Azure account mismatch: credential broker configured {expected_account}, path {account}"
        )));
    }
    if suffix != expected_suffix {
        return Err(invalid(format!(
            "Azure endpoint suffix mismatch: configured {expected_suffix}, path {suffix}"
        )));
    }

    Ok(AzureLocation {
        account: account.to_string(),
        container: container.to_string(),
        endpoint: format!("https://{account}.blob.{suffix}"),
        path: Path::from_url_path(url.path())
            .map_err(|error| invalid(format!("Invalid Azure object path: {error}")))?,
    })
}

fn parse_gcp_location(absolute_path: &str) -> IcebergResult<(String, Path)> {
    let url = Url::parse(absolute_path)
        .map_err(|error| invalid(format!("Invalid GCS URL {absolute_path}: {error}")))?;
    if !matches!(url.scheme(), "gs" | "gcs") {
        return Err(invalid(format!("Unsupported GCS scheme: {}", url.scheme())));
    }
    let bucket = url
        .host_str()
        .filter(|bucket| !bucket.is_empty())
        .ok_or_else(|| invalid(format!("GCS URL is missing a bucket: {absolute_path}")))?;
    let path = Path::from_url_path(url.path())
        .map_err(|error| invalid(format!("Invalid GCS object path: {error}")))?;
    Ok((bucket.to_string(), path))
}

fn invalid(message: impl Into<String>) -> IcebergError {
    IcebergError::new(IcebergErrorKind::DataInvalid, message.into())
}

fn unexpected(
    message: &'static str,
    source: impl std::error::Error + Send + Sync + 'static,
) -> IcebergError {
    IcebergError::new(IcebergErrorKind::Unexpected, message).with_source(source)
}

fn from_object_store_error(error: object_store::Error) -> IcebergError {
    unexpected("object store operation failed", error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_normalized_azure_location() {
        let location = parse_azure_location(
            "abfss://container@account.dfs.core.windows.net/dir/file.json",
            "account",
            "container",
            "core.windows.net",
        )
        .unwrap();
        assert_eq!(location.account, "account");
        assert_eq!(location.container, "container");
        assert_eq!(location.endpoint, "https://account.blob.core.windows.net");
        assert_eq!(location.path.as_ref(), "dir/file.json");
    }

    #[test]
    fn rejects_azure_account_mismatch() {
        assert!(
            parse_azure_location(
                "abfss://container@account.dfs.core.windows.net/file.json",
                "other",
                "container",
                "core.windows.net",
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_azure_container_mismatch() {
        let result = parse_azure_location(
            "abfss://other@account.dfs.core.windows.net/file.json",
            "account",
            "container",
            "core.windows.net",
        );
        let error = match result {
            Ok(_) => panic!("expected Azure container mismatch"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("Azure container mismatch"));
    }

    #[test]
    fn rejects_azure_endpoint_suffix_mismatch() {
        assert!(
            parse_azure_location(
                "abfss://container@account.dfs.attacker.example/file.json",
                "account",
                "container",
                "core.windows.net",
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_insecure_azure_schemes() {
        for scheme in ["abfs", "wasb"] {
            assert!(
                parse_azure_location(
                    &format!(
                        "{scheme}://container@account.{}.core.windows.net/file.json",
                        if scheme == "abfs" { "dfs" } else { "blob" }
                    ),
                    "account",
                    "container",
                    "core.windows.net",
                )
                .is_err()
            );
        }
    }

    #[test]
    fn parses_gcp_location() {
        let (bucket, path) = parse_gcp_location("gs://bucket/dir/file%20name.json").unwrap();
        assert_eq!(bucket, "bucket");
        assert_eq!(path.as_ref(), "dir/file name.json");
    }
}
