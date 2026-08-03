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
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use anyhow::{Result as AnyResult, bail};
use lance::{Error as LanceError, Result as LanceResult};
use lru::LruCache;
use tokio::sync::{Mutex, OnceCell};

use crate::aliyun_oss_provider::RefreshableAliyunOssStore;
use crate::azure_sas_provider::{AzureBrokerConfig, AzureSasStorageOptionsProvider};
use crate::gcp_impersonation::ImpersonatingGcsCredentialProvider;

pub(crate) const CACHE_KEY: &str = "milvus_fs_cache_key";

// Credential providers are small, but each one retains refresh state and HTTP
// clients. Bound the process-global registry so arbitrary external filesystem
// churn cannot grow it forever. Eviction is safe: active readers retain Arcs.
const MAX_CACHED_PROVIDERS: usize = 1024;

#[derive(Clone)]
struct CacheEntry<C, V> {
    config: C,
    value: Arc<OnceCell<V>>,
}

fn new_cache<K, C, V>() -> Mutex<LruCache<K, CacheEntry<C, V>>>
where
    K: Eq + Hash,
{
    Mutex::new(LruCache::new(
        NonZeroUsize::new(MAX_CACHED_PROVIDERS).expect("credential cache capacity is non-zero"),
    ))
}

async fn cached_cell<K, C, V>(
    cache: &Mutex<LruCache<K, CacheEntry<C, V>>>,
    key: K,
    config: &C,
    provider_name: &str,
) -> Result<Arc<OnceCell<V>>, String>
where
    K: Clone + Eq + Hash,
    C: Clone + Eq,
{
    let mut cache = cache.lock().await;
    if let Some(entry) = cache.get(&key) {
        if &entry.config != config {
            return Err(format!(
                "{provider_name} credential cache key was reused with different configuration"
            ));
        }
        return Ok(entry.value.clone());
    }

    let value = Arc::new(OnceCell::new());
    cache.put(
        key,
        CacheEntry {
            config: config.clone(),
            value: value.clone(),
        },
    );
    Ok(value)
}

fn usable_cache_key(cache_key: Option<&str>) -> Option<&str> {
    cache_key.filter(|key| !key.is_empty())
}

/// Configuration for AWS STS AssumeRole credentials.
#[derive(Clone, Eq, PartialEq)]
pub(crate) struct AssumeRoleConfig {
    role_arn: String,
    region: String,
    session_name: String,
    external_id: String,
    credential_refresh_secs: u64,
}

impl AssumeRoleConfig {
    /// Parse from raw parameters. Returns None if role_arn is empty.
    /// Returns Err if credential_refresh_secs is out of range [900, 43200].
    /// 43200s (12h) is AWS STS `AssumeRole`'s hard upper bound on
    /// `DurationSeconds`, reachable only when the target IAM role's
    /// `MaxSessionDuration` is raised from the 3600s default.
    pub(crate) fn parse(
        role_arn: &str,
        region: &str,
        session_name: &str,
        external_id: &str,
        credential_refresh_secs: u64,
    ) -> LanceResult<Option<Self>> {
        if role_arn.is_empty() {
            return Ok(None);
        }
        if !(900..=43200).contains(&credential_refresh_secs) {
            return Err(LanceError::invalid_input(format!(
                "credential_refresh_secs must be in [900, 43200], got {credential_refresh_secs}"
            )));
        }
        Ok(Some(Self {
            role_arn: role_arn.to_string(),
            region: region.to_string(),
            session_name: session_name.to_string(),
            external_id: external_id.to_string(),
            credential_refresh_secs,
        }))
    }

    async fn build_credentials(&self) -> LanceResult<object_store::aws::AwsCredentialProvider> {
        use aws_config::sts::AssumeRoleProvider;
        use lance_io::object_store::providers::aws::AwsCredentialAdapter;

        // Must remain strictly below the minimum accepted session length, or
        // the adapter would consider every freshly-issued credential stale.
        const REFRESH_OFFSET_SECS: u64 = 300;

        let mut builder = AssumeRoleProvider::builder(&self.role_arn)
            .session_length(Duration::from_secs(self.credential_refresh_secs));
        if !self.region.is_empty() {
            let region = aws_config::Region::new(self.region.clone());
            // Configure the default source-credential chain with the same
            // region before constructing the outer AssumeRole provider. A
            // region override on the outer STS client alone is too late for
            // WebIdentity/IRSA or profile-based source credentials.
            let sdk_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
                .region(region.clone())
                .load()
                .await;
            builder = builder.configure(&sdk_config).region(region);
        }
        if !self.session_name.is_empty() {
            builder = builder.session_name(&self.session_name);
        }
        if !self.external_id.is_empty() {
            builder = builder.external_id(&self.external_id);
        }

        let assume_role_provider = builder.build().await;
        Ok(Arc::new(AwsCredentialAdapter::new(
            Arc::new(assume_role_provider),
            Duration::from_secs(REFRESH_OFFSET_SECS),
        )))
    }
}

type AwsProvider = object_store::aws::AwsCredentialProvider;
static AWS_PROVIDERS: LazyLock<Mutex<LruCache<String, CacheEntry<AssumeRoleConfig, AwsProvider>>>> =
    LazyLock::new(new_cache);

pub(crate) async fn cached_aws_credentials(
    cache_key: Option<&str>,
    config: &AssumeRoleConfig,
) -> LanceResult<AwsProvider> {
    let Some(cache_key) = usable_cache_key(cache_key) else {
        return config.build_credentials().await;
    };

    let cell = cached_cell(&AWS_PROVIDERS, cache_key.to_string(), config, "AWS")
        .await
        .map_err(LanceError::invalid_input)?;
    cell.get_or_try_init(|| config.build_credentials())
        .await
        .cloned()
}

type AzureProvider = Arc<AzureSasStorageOptionsProvider>;
static AZURE_PROVIDERS: LazyLock<
    Mutex<LruCache<String, CacheEntry<AzureBrokerConfig, AzureProvider>>>,
> = LazyLock::new(new_cache);

pub(crate) async fn cached_azure_sas_provider(
    cache_key: Option<&str>,
    config: AzureBrokerConfig,
) -> AnyResult<AzureProvider> {
    let Some(cache_key) = usable_cache_key(cache_key) else {
        return Ok(Arc::new(AzureSasStorageOptionsProvider::new(config)?));
    };

    let cell = cached_cell(&AZURE_PROVIDERS, cache_key.to_string(), &config, "Azure")
        .await
        .map_err(anyhow::Error::msg)?;
    cell.get_or_try_init(
        || async move { Ok(Arc::new(AzureSasStorageOptionsProvider::new(config)?)) },
    )
    .await
    .cloned()
}

#[derive(Clone, Eq, PartialEq)]
struct GcpProviderConfig {
    target_sa: String,
    token_lifetime: Duration,
    refresh_offset: Duration,
}

type GcpProvider = Arc<ImpersonatingGcsCredentialProvider>;
static GCP_PROVIDERS: LazyLock<
    Mutex<LruCache<String, CacheEntry<GcpProviderConfig, GcpProvider>>>,
> = LazyLock::new(new_cache);

pub(crate) async fn cached_gcp_credential_provider(
    cache_key: Option<&str>,
    target_sa: String,
    token_lifetime: Duration,
    refresh_offset: Duration,
) -> AnyResult<GcpProvider> {
    let config = GcpProviderConfig {
        target_sa,
        token_lifetime,
        refresh_offset,
    };
    let build = || {
        Arc::new(ImpersonatingGcsCredentialProvider::new(
            config.target_sa.clone(),
            config.token_lifetime,
            config.refresh_offset,
        ))
    };

    let Some(cache_key) = usable_cache_key(cache_key) else {
        return Ok(build());
    };
    let cell = cached_cell(&GCP_PROVIDERS, cache_key.to_string(), &config, "GCP")
        .await
        .map_err(anyhow::Error::msg)?;
    Ok(cell.get_or_init(|| async { build() }).await.clone())
}

#[derive(Clone, Eq, PartialEq)]
struct AliyunProviderConfig {
    base_config: HashMap<String, String>,
    refresh_interval: Duration,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct AliyunCacheKey {
    filesystem_key: String,
    bucket: String,
}

type AliyunProvider = Arc<RefreshableAliyunOssStore>;
static ALIYUN_PROVIDERS: LazyLock<
    Mutex<LruCache<AliyunCacheKey, CacheEntry<AliyunProviderConfig, AliyunProvider>>>,
> = LazyLock::new(new_cache);

pub(crate) async fn cached_aliyun_oss_store(
    cache_key: Option<&str>,
    bucket: &str,
    base_config: HashMap<String, String>,
    refresh_interval: Duration,
) -> AnyResult<AliyunProvider> {
    let config = AliyunProviderConfig {
        base_config,
        refresh_interval,
    };
    let build = || {
        Arc::new(RefreshableAliyunOssStore::new(
            config.base_config.clone(),
            config.refresh_interval,
        ))
    };

    let Some(cache_key) = usable_cache_key(cache_key) else {
        return Ok(build());
    };
    if bucket.is_empty() {
        bail!("Aliyun credential cache bucket must not be empty");
    }
    let key = AliyunCacheKey {
        filesystem_key: cache_key.to_string(),
        bucket: bucket.to_string(),
    };
    let cell = cached_cell(&ALIYUN_PROVIDERS, key, &config, "Aliyun")
        .await
        .map_err(anyhow::Error::msg)?;
    Ok(cell.get_or_init(|| async { build() }).await.clone())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn assume_role_config_validates_session_lifetime() {
        assert!(AssumeRoleConfig::parse("arn", "us-west-2", "session", "", 899).is_err());
        assert!(AssumeRoleConfig::parse("arn", "us-west-2", "session", "", 900).is_ok());
        assert!(AssumeRoleConfig::parse("arn", "us-west-2", "session", "", 43_200).is_ok());
        assert!(AssumeRoleConfig::parse("arn", "us-west-2", "session", "", 43_201).is_err());
        assert!(
            AssumeRoleConfig::parse("", "us-west-2", "session", "", 0)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn assume_role_config_includes_region_in_cache_identity() {
        let us = AssumeRoleConfig::parse("arn", "us-west-2", "session", "", 900)
            .unwrap()
            .unwrap();
        let cn = AssumeRoleConfig::parse("arn", "cn-north-1", "session", "", 900)
            .unwrap()
            .unwrap();
        assert!(us != cn);
        assert_eq!(cn.region, "cn-north-1");
    }

    #[tokio::test]
    async fn cache_singleflights_concurrent_initialization() {
        let cache = Arc::new(new_cache::<String, u8, usize>());
        let starts = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..16 {
            let cache = cache.clone();
            let starts = starts.clone();
            tasks.push(tokio::spawn(async move {
                let cell = cached_cell(&cache, "shared".to_string(), &7, "test")
                    .await
                    .unwrap();
                *cell
                    .get_or_init(|| async move {
                        starts.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        42
                    })
                    .await
            }));
        }

        for task in tasks {
            assert_eq!(task.await.unwrap(), 42);
        }
        assert_eq!(starts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cache_rejects_same_key_with_different_configuration() {
        let cache = new_cache::<String, u8, usize>();
        cached_cell(&cache, "shared".to_string(), &1, "test")
            .await
            .unwrap();
        let error = cached_cell(&cache, "shared".to_string(), &2, "test")
            .await
            .unwrap_err();
        assert!(error.contains("different configuration"));
    }

    #[tokio::test]
    async fn gcp_reuses_only_matching_non_empty_keys() {
        let duration = Duration::from_secs(900);
        let offset = Duration::from_secs(300);
        let first = cached_gcp_credential_provider(
            Some("test-gcp-reuse"),
            "one@example.com".to_string(),
            duration,
            offset,
        )
        .await
        .unwrap();
        let same = cached_gcp_credential_provider(
            Some("test-gcp-reuse"),
            "one@example.com".to_string(),
            duration,
            offset,
        )
        .await
        .unwrap();
        assert!(Arc::ptr_eq(&first, &same));

        let other_key = cached_gcp_credential_provider(
            Some("test-gcp-isolation"),
            "one@example.com".to_string(),
            duration,
            offset,
        )
        .await
        .unwrap();
        assert!(!Arc::ptr_eq(&first, &other_key));

        let mismatch = cached_gcp_credential_provider(
            Some("test-gcp-reuse"),
            "two@example.com".to_string(),
            duration,
            offset,
        )
        .await;
        assert!(mismatch.is_err());

        let uncached_one =
            cached_gcp_credential_provider(None, "one@example.com".to_string(), duration, offset)
                .await
                .unwrap();
        let uncached_two = cached_gcp_credential_provider(
            Some(""),
            "one@example.com".to_string(),
            duration,
            offset,
        )
        .await
        .unwrap();
        assert!(!Arc::ptr_eq(&uncached_one, &uncached_two));
    }

    #[tokio::test]
    async fn azure_reuses_matching_key_and_rejects_configuration_mismatch() {
        let config = |client_id: &str| {
            let mut options = HashMap::from([
                (
                    "azure_broker_endpoint".to_string(),
                    "http://broker".to_string(),
                ),
                ("azure_broker_client_id".to_string(), client_id.to_string()),
                ("azure_broker_tenant_id".to_string(), "tenant".to_string()),
                (
                    "azure_broker_account_name".to_string(),
                    "account".to_string(),
                ),
                ("azure_broker_region".to_string(), "westus3".to_string()),
                ("azure_broker_bucket".to_string(), "container".to_string()),
                (
                    "azure_broker_duration_seconds".to_string(),
                    "3600".to_string(),
                ),
                (
                    "azure_broker_request_timeout_ms".to_string(),
                    "1000".to_string(),
                ),
            ]);
            AzureBrokerConfig::extract(&mut options).unwrap().unwrap()
        };

        let first = cached_azure_sas_provider(Some("test-azure-reuse"), config("client-a"))
            .await
            .unwrap();
        let same = cached_azure_sas_provider(Some("test-azure-reuse"), config("client-a"))
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&first, &same));

        let other_key = cached_azure_sas_provider(Some("test-azure-isolation"), config("client-a"))
            .await
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &other_key));

        let mismatch =
            cached_azure_sas_provider(Some("test-azure-reuse"), config("client-b")).await;
        assert!(mismatch.is_err());
    }

    #[tokio::test]
    async fn aliyun_store_cache_is_bucket_scoped() {
        let config = |bucket: &str| HashMap::from([("bucket".to_string(), bucket.to_string())]);
        let first = cached_aliyun_oss_store(
            Some("test-aliyun-bucket"),
            "bucket-a",
            config("bucket-a"),
            Duration::from_secs(900),
        )
        .await
        .unwrap();
        let same = cached_aliyun_oss_store(
            Some("test-aliyun-bucket"),
            "bucket-a",
            config("bucket-a"),
            Duration::from_secs(900),
        )
        .await
        .unwrap();
        let other_bucket = cached_aliyun_oss_store(
            Some("test-aliyun-bucket"),
            "bucket-b",
            config("bucket-b"),
            Duration::from_secs(900),
        )
        .await
        .unwrap();

        assert!(Arc::ptr_eq(&first, &same));
        assert!(!Arc::ptr_eq(&first, &other_bucket));
    }
}
