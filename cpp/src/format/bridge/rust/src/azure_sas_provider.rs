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
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result as AnyResult, anyhow, bail};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lance_core::error::{Error as LanceError, Result as LanceResult};
use lance_io::object_store::StorageOptionsProvider;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

pub(crate) const AZURE_BROKER_ENDPOINT: &str = "azure_broker_endpoint";
pub(crate) const AZURE_BROKER_CLIENT_ID: &str = "azure_broker_client_id";
pub(crate) const AZURE_BROKER_TENANT_ID: &str = "azure_broker_tenant_id";
pub(crate) const AZURE_BROKER_ACCOUNT_NAME: &str = "azure_broker_account_name";
pub(crate) const AZURE_BROKER_REGION: &str = "azure_broker_region";
pub(crate) const AZURE_BROKER_BUCKET: &str = "azure_broker_bucket";
pub(crate) const AZURE_BROKER_DURATION_SECONDS: &str = "azure_broker_duration_seconds";
pub(crate) const AZURE_BROKER_REQUEST_TIMEOUT_MS: &str = "azure_broker_request_timeout_ms";

const REFRESH_OFFSET_SECONDS: i64 = 60;

// These options are private to the C++/Rust bridge. They must be consumed
// before the remaining storage options are passed to Lance or OpenDAL.
const BROKER_KEYS: [&str; 8] = [
    AZURE_BROKER_ENDPOINT,
    AZURE_BROKER_CLIENT_ID,
    AZURE_BROKER_TENANT_ID,
    AZURE_BROKER_ACCOUNT_NAME,
    AZURE_BROKER_REGION,
    AZURE_BROKER_BUCKET,
    AZURE_BROKER_DURATION_SECONDS,
    AZURE_BROKER_REQUEST_TIMEOUT_MS,
];

/// Typed Azure credential broker configuration produced from bridge-private
/// storage options populated by the C++ Lance and Iceberg adapters.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub(crate) struct AzureBrokerConfig {
    endpoint: String,
    client_id: String,
    tenant_id: String,
    account_name: String,
    region: String,
    bucket: String,
    duration_seconds: u64,
    request_timeout_ms: u64,
}

impl AzureBrokerConfig {
    /// Removes and parses the broker options from `options`.
    ///
    /// No broker key means that broker authentication is disabled. Once any
    /// broker key is present, the configuration is treated as all-or-nothing so
    /// a partial setup cannot silently fall back to another Azure auth mode.
    pub(crate) fn extract(options: &mut HashMap<String, String>) -> AnyResult<Option<Self>> {
        let enabled = BROKER_KEYS.iter().any(|key| options.contains_key(*key));
        if !enabled {
            return Ok(None);
        }

        let mut take = |key: &str| options.remove(key).unwrap_or_default();
        let config = Self {
            endpoint: take(AZURE_BROKER_ENDPOINT),
            client_id: take(AZURE_BROKER_CLIENT_ID),
            tenant_id: take(AZURE_BROKER_TENANT_ID),
            account_name: take(AZURE_BROKER_ACCOUNT_NAME),
            region: take(AZURE_BROKER_REGION),
            bucket: take(AZURE_BROKER_BUCKET),
            duration_seconds: take(AZURE_BROKER_DURATION_SECONDS).parse().unwrap_or(0),
            request_timeout_ms: take(AZURE_BROKER_REQUEST_TIMEOUT_MS).parse().unwrap_or(0),
        };

        if config.endpoint.is_empty()
            || config.client_id.is_empty()
            || config.tenant_id.is_empty()
            || config.account_name.is_empty()
            || config.region.is_empty()
            || config.bucket.is_empty()
            || config.duration_seconds == 0
            || config.request_timeout_ms == 0
        {
            bail!("incomplete Azure credential broker configuration");
        }

        let endpoint = url::Url::parse(&config.endpoint)
            .map_err(|_| anyhow!("Azure credential broker endpoint is not a valid URL"))?;
        if (endpoint.scheme() != "http" && endpoint.scheme() != "https") || !endpoint.has_host() {
            bail!("Azure credential broker endpoint must use HTTP or HTTPS");
        }

        Ok(Some(config))
    }

    pub(crate) fn bucket(&self) -> &str {
        &self.bucket
    }

    pub(crate) fn account_name(&self) -> &str {
        &self.account_name
    }

    fn provider_id(&self) -> String {
        // Lance uses this opaque ID to distinguish providers without embedding
        // the raw broker configuration in the identifier.
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        format!("azure_sas_broker_{:016x}", hasher.finish())
    }
}

#[derive(Clone)]
pub(crate) struct AzureSasCredential {
    pub(crate) token: String,
    pub(crate) expires_at: DateTime<Utc>,
}

#[derive(Clone)]
pub(crate) struct AzureBrokerClient {
    config: AzureBrokerConfig,
    client: reqwest::Client,
}

impl fmt::Debug for AzureBrokerClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureBrokerClient")
            .field("provider_id", &self.config.provider_id())
            .finish()
    }
}

#[derive(Serialize)]
struct BrokerRequest<'a> {
    csp: &'static str,
    region: &'a str,
    bucket: &'a str,
    #[serde(rename = "durationSeconds")]
    duration_seconds: u64,
    #[serde(rename = "azureClientId")]
    azure_client_id: &'a str,
    #[serde(rename = "azureTenantId")]
    azure_tenant_id: &'a str,
    #[serde(rename = "azureAccountName")]
    azure_account_name: &'a str,
}

#[derive(Deserialize)]
struct BrokerResponse {
    success: bool,
    credentials: Option<BrokerCredentials>,
}

#[derive(Deserialize)]
struct BrokerCredentials {
    #[serde(rename = "tempAk")]
    temp_ak: String,
    #[serde(rename = "sessionToken")]
    session_token: String,
    #[serde(rename = "expiredAt")]
    expired_at: String,
}

impl AzureBrokerClient {
    pub(crate) fn new(config: AzureBrokerConfig) -> AnyResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.request_timeout_ms))
            .build()
            .map_err(|_| anyhow!("failed to construct Azure credential broker HTTP client"))?;
        Ok(Self { config, client })
    }

    pub(crate) async fn fetch(&self, now: DateTime<Utc>) -> AnyResult<AzureSasCredential> {
        let request = BrokerRequest {
            csp: "azure",
            region: &self.config.region,
            bucket: &self.config.bucket,
            duration_seconds: self.config.duration_seconds,
            azure_client_id: &self.config.client_id,
            azure_tenant_id: &self.config.tenant_id,
            azure_account_name: &self.config.account_name,
        };
        let response = self
            .client
            .post(&self.config.endpoint)
            .json(&request)
            .send()
            .await
            .map_err(|_| anyhow!("transport_error"))?;
        let status = response.status();
        if !status.is_success() {
            bail!("http_status={}", status.as_u16());
        }
        let response: BrokerResponse =
            response.json().await.map_err(|_| anyhow!("invalid_json"))?;
        if !response.success {
            bail!("business_failure");
        }
        let credentials = response
            .credentials
            .ok_or_else(|| anyhow!("missing_credentials"))?;
        if credentials.temp_ak != self.config.account_name {
            bail!("account_mismatch");
        }
        let token = credentials
            .session_token
            .trim_start_matches('?')
            .to_string();
        if token.is_empty() {
            bail!("empty_sas");
        }
        if !url::form_urlencoded::parse(token.as_bytes())
            .any(|(key, value)| key == "sig" && !value.is_empty())
        {
            bail!("missing_sas_signature");
        }
        let expires_at = DateTime::parse_from_rfc3339(&credentials.expired_at)
            .map_err(|_| anyhow!("invalid_expiration"))?
            .with_timezone(&Utc);
        if expires_at <= now {
            bail!("expired_credential");
        }
        Ok(AzureSasCredential { token, expires_at })
    }
}

#[async_trait]
trait AzureSasFetcher: Send + Sync {
    async fn fetch(&self, now: DateTime<Utc>) -> AnyResult<AzureSasCredential>;
}

#[async_trait]
impl AzureSasFetcher for AzureBrokerClient {
    async fn fetch(&self, now: DateTime<Utc>) -> AnyResult<AzureSasCredential> {
        AzureBrokerClient::fetch(self, now).await
    }
}

type Clock = Arc<dyn Fn() -> DateTime<Utc> + Send + Sync>;

pub(crate) struct AzureSasStorageOptionsProvider {
    provider_id: String,
    fetcher: Arc<dyn AzureSasFetcher>,
    clock: Clock,
    cached: RwLock<Option<AzureSasCredential>>,
}

impl fmt::Debug for AzureSasStorageOptionsProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AzureSasStorageOptionsProvider")
            .field("provider_id", &self.provider_id)
            .field(
                "has_cached_sas",
                &self.cached.try_read().map(|v| v.is_some()).unwrap_or(false),
            )
            .finish()
    }
}

impl AzureSasStorageOptionsProvider {
    pub(crate) fn new(config: AzureBrokerConfig) -> AnyResult<Self> {
        let provider_id = config.provider_id();
        let fetcher = Arc::new(AzureBrokerClient::new(config)?);
        Ok(Self {
            provider_id,
            fetcher,
            clock: Arc::new(Utc::now),
            cached: RwLock::new(None),
        })
    }

    #[cfg(test)]
    fn with_fetcher(
        config: AzureBrokerConfig,
        fetcher: Arc<dyn AzureSasFetcher>,
        clock: Clock,
    ) -> Self {
        Self {
            provider_id: config.provider_id(),
            fetcher,
            clock,
            cached: RwLock::new(None),
        }
    }

    fn is_fresh(credential: &AzureSasCredential, now: DateTime<Utc>) -> bool {
        credential.expires_at - now > chrono::Duration::seconds(REFRESH_OFFSET_SECONDS)
    }

    fn to_options(credential: &AzureSasCredential) -> HashMap<String, String> {
        HashMap::from([
            (
                "azure_storage_sas_token".to_string(),
                credential.token.clone(),
            ),
            (
                "expires_at_millis".to_string(),
                credential.expires_at.timestamp_millis().to_string(),
            ),
        ])
    }

    fn lance_error(error: &anyhow::Error) -> LanceError {
        LanceError::io_source(Box::new(std::io::Error::other(format!(
            "Azure SAS credential broker failure: {error}"
        ))))
    }
}

#[async_trait]
impl StorageOptionsProvider for AzureSasStorageOptionsProvider {
    async fn fetch_storage_options(&self) -> LanceResult<Option<HashMap<String, String>>> {
        let mut now = (self.clock)();
        {
            let cached = self.cached.read().await;
            if let Some(credential) = cached.as_ref()
                && Self::is_fresh(credential, now)
            {
                return Ok(Some(Self::to_options(credential)));
            }
        }

        let mut cached = self.cached.write().await;
        now = (self.clock)();
        if let Some(credential) = cached.as_ref()
            && Self::is_fresh(credential, now)
        {
            return Ok(Some(Self::to_options(credential)));
        }

        let refresh_result = self.fetcher.fetch(now).await;
        now = (self.clock)();
        let refresh_result = refresh_result.and_then(|credential| {
            if credential.expires_at <= now {
                Err(anyhow!("expired_credential"))
            } else {
                Ok(credential)
            }
        });

        match refresh_result {
            Ok(credential) => {
                let options = Self::to_options(&credential);
                *cached = Some(credential);
                Ok(Some(options))
            }
            Err(error) => {
                let has_cached_sas = cached.is_some();
                let cached_expired = cached
                    .as_ref()
                    .map(|credential| credential.expires_at <= now)
                    .unwrap_or(false);
                eprintln!(
                    "Warning: Azure SAS credential broker refresh failed: {}, has_cached_sas={}, cached_expired={}",
                    error, has_cached_sas, cached_expired
                );
                if let Some(credential) = cached
                    .as_ref()
                    .filter(|credential| credential.expires_at > now)
                {
                    Ok(Some(Self::to_options(credential)))
                } else {
                    Err(Self::lance_error(&error))
                }
            }
        }
    }

    fn provider_id(&self) -> String {
        self.provider_id.clone()
    }
}

#[async_trait]
impl object_store::CredentialProvider for AzureSasStorageOptionsProvider {
    type Credential = object_store::azure::AzureCredential;

    async fn get_credential(
        &self,
    ) -> object_store::Result<Arc<object_store::azure::AzureCredential>> {
        let options = <Self as StorageOptionsProvider>::fetch_storage_options(self)
            .await
            .map_err(|error| object_store::Error::Generic {
                store: "azure_sas_broker",
                source: Box::new(std::io::Error::other(error.to_string())),
            })?
            .ok_or_else(|| object_store::Error::Generic {
                store: "azure_sas_broker",
                source: Box::new(std::io::Error::other(
                    "Azure SAS provider returned no credential",
                )),
            })?;
        let token =
            options
                .get("azure_storage_sas_token")
                .ok_or_else(|| object_store::Error::Generic {
                    store: "azure_sas_broker",
                    source: Box::new(std::io::Error::other(
                        "Azure SAS provider returned no token",
                    )),
                })?;
        let query_pairs = url::form_urlencoded::parse(token.as_bytes())
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect();
        Ok(Arc::new(object_store::azure::AzureCredential::SASToken(
            query_pairs,
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    fn config() -> AzureBrokerConfig {
        AzureBrokerConfig {
            endpoint: "http://credential-broker/v1/credentials/assume-role".to_string(),
            client_id: "client".to_string(),
            tenant_id: "tenant".to_string(),
            account_name: "account".to_string(),
            region: "westus3".to_string(),
            bucket: "container".to_string(),
            duration_seconds: 3600,
            request_timeout_ms: 1000,
        }
    }

    struct MockFetcher {
        responses: Mutex<VecDeque<Result<AzureSasCredential, &'static str>>>,
        calls: AtomicUsize,
    }

    impl MockFetcher {
        fn new(responses: Vec<Result<AzureSasCredential, &'static str>>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl AzureSasFetcher for MockFetcher {
        async fn fetch(&self, _now: DateTime<Utc>) -> AnyResult<AzureSasCredential> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(Err("no_response"))
                .map_err(|error| anyhow!(error))
        }
    }

    struct ClockAdvancingFetcher {
        responses: Mutex<VecDeque<(chrono::Duration, Result<AzureSasCredential, &'static str>)>>,
        clock: Arc<Mutex<DateTime<Utc>>>,
    }

    #[async_trait]
    impl AzureSasFetcher for ClockAdvancingFetcher {
        async fn fetch(&self, _now: DateTime<Utc>) -> AnyResult<AzureSasCredential> {
            let (advance, response) = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("missing fetch response");
            *self.clock.lock().unwrap() += advance;
            response.map_err(|error| anyhow!(error))
        }
    }

    #[test]
    fn extracts_and_removes_private_options() {
        let mut options = HashMap::from([
            (AZURE_BROKER_ENDPOINT.to_string(), config().endpoint),
            (AZURE_BROKER_CLIENT_ID.to_string(), "client".to_string()),
            (AZURE_BROKER_TENANT_ID.to_string(), "tenant".to_string()),
            (AZURE_BROKER_ACCOUNT_NAME.to_string(), "account".to_string()),
            (AZURE_BROKER_REGION.to_string(), "westus3".to_string()),
            (AZURE_BROKER_BUCKET.to_string(), "container".to_string()),
            (
                AZURE_BROKER_DURATION_SECONDS.to_string(),
                "3600".to_string(),
            ),
            (
                AZURE_BROKER_REQUEST_TIMEOUT_MS.to_string(),
                "1000".to_string(),
            ),
            (
                "azure_storage_account_name".to_string(),
                "account".to_string(),
            ),
        ]);

        let extracted = AzureBrokerConfig::extract(&mut options).unwrap().unwrap();
        assert_eq!(extracted, config());
        assert!(BROKER_KEYS.iter().all(|key| !options.contains_key(*key)));
        assert_eq!(options["azure_storage_account_name"], "account");
    }

    #[test]
    fn rejects_partial_or_non_http_configuration() {
        let mut partial = HashMap::from([(
            AZURE_BROKER_ENDPOINT.to_string(),
            "http://credential-broker".to_string(),
        )]);
        assert!(AzureBrokerConfig::extract(&mut partial).is_err());

        let mut invalid = HashMap::from([
            (
                AZURE_BROKER_ENDPOINT.to_string(),
                "file:///tmp/token".to_string(),
            ),
            (AZURE_BROKER_CLIENT_ID.to_string(), "client".to_string()),
            (AZURE_BROKER_TENANT_ID.to_string(), "tenant".to_string()),
            (AZURE_BROKER_ACCOUNT_NAME.to_string(), "account".to_string()),
            (AZURE_BROKER_REGION.to_string(), "westus3".to_string()),
            (AZURE_BROKER_BUCKET.to_string(), "container".to_string()),
            (
                AZURE_BROKER_DURATION_SECONDS.to_string(),
                "3600".to_string(),
            ),
            (
                AZURE_BROKER_REQUEST_TIMEOUT_MS.to_string(),
                "1000".to_string(),
            ),
        ]);
        assert!(AzureBrokerConfig::extract(&mut invalid).is_err());
    }

    #[tokio::test]
    async fn rejects_sas_without_non_empty_signature() {
        for token in ["sv=1", "sv=1&sig="] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let now = Utc::now();
            let response_body = serde_json::json!({
                "success": true,
                "credentials": {
                    "tempAk": "account",
                    "sessionToken": token,
                    "expiredAt": (now + chrono::Duration::hours(1)).to_rfc3339(),
                }
            })
            .to_string();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = [0_u8; 4096];
                socket.read(&mut request).await.unwrap();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            });

            let mut broker_config = config();
            broker_config.endpoint = format!("http://{address}");
            let client = AzureBrokerClient::new(broker_config).unwrap();
            let error = match client.fetch(now).await {
                Ok(_) => panic!("expected SAS signature validation failure"),
                Err(error) => error,
            };
            assert_eq!(error.to_string(), "missing_sas_signature");
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn caches_and_retries_failed_refresh_with_old_token() {
        let now = Arc::new(Mutex::new(Utc::now()));
        let initial_now = *now.lock().unwrap();
        let fetcher = Arc::new(MockFetcher::new(vec![
            Ok(AzureSasCredential {
                token: "sv=1&sig=old".to_string(),
                expires_at: initial_now + chrono::Duration::seconds(120),
            }),
            Err("http_status=500"),
            Err("http_status=500"),
            Ok(AzureSasCredential {
                token: "sv=2&sig=new".to_string(),
                expires_at: initial_now + chrono::Duration::hours(2),
            }),
        ]));
        let clock_now = now.clone();
        let provider = AzureSasStorageOptionsProvider::with_fetcher(
            config(),
            fetcher.clone(),
            Arc::new(move || *clock_now.lock().unwrap()),
        );

        let first = provider.fetch_storage_options().await.unwrap().unwrap();
        assert_eq!(first["azure_storage_sas_token"], "sv=1&sig=old");

        *now.lock().unwrap() += chrono::Duration::seconds(61);
        let fallback1 = provider.fetch_storage_options().await.unwrap().unwrap();
        let fallback2 = provider.fetch_storage_options().await.unwrap().unwrap();
        assert_eq!(fallback1["azure_storage_sas_token"], "sv=1&sig=old");
        assert_eq!(fallback2["azure_storage_sas_token"], "sv=1&sig=old");
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 3);

        *now.lock().unwrap() += chrono::Duration::seconds(60);
        let refreshed = provider.fetch_storage_options().await.unwrap().unwrap();
        assert_eq!(refreshed["azure_storage_sas_token"], "sv=2&sig=new");
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn fails_closed_without_cached_token() {
        let now = Utc::now();
        let fetcher = Arc::new(MockFetcher::new(vec![Err("transport_error")]));
        let provider =
            AzureSasStorageOptionsProvider::with_fetcher(config(), fetcher, Arc::new(move || now));
        assert!(provider.fetch_storage_options().await.is_err());
    }

    #[tokio::test]
    async fn fails_closed_when_cached_token_is_expired() {
        let now = Arc::new(Mutex::new(Utc::now()));
        let initial_now = *now.lock().unwrap();
        let fetcher = Arc::new(MockFetcher::new(vec![
            Ok(AzureSasCredential {
                token: "sv=1&sig=old".to_string(),
                expires_at: initial_now + chrono::Duration::seconds(120),
            }),
            Err("transport_error"),
        ]));
        let clock_now = now.clone();
        let provider = AzureSasStorageOptionsProvider::with_fetcher(
            config(),
            fetcher.clone(),
            Arc::new(move || *clock_now.lock().unwrap()),
        );

        assert!(provider.fetch_storage_options().await.is_ok());
        *now.lock().unwrap() += chrono::Duration::seconds(120);

        assert!(provider.fetch_storage_options().await.is_err());
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn fails_closed_when_cached_token_expires_during_refresh() {
        let now = Arc::new(Mutex::new(Utc::now()));
        let initial_now = *now.lock().unwrap();
        let fetcher = Arc::new(ClockAdvancingFetcher {
            responses: Mutex::new(
                vec![
                    (
                        chrono::Duration::zero(),
                        Ok(AzureSasCredential {
                            token: "sv=1&sig=old".to_string(),
                            expires_at: initial_now + chrono::Duration::seconds(120),
                        }),
                    ),
                    (chrono::Duration::seconds(60), Err("transport_error")),
                ]
                .into(),
            ),
            clock: now.clone(),
        });
        let clock_now = now.clone();
        let provider = AzureSasStorageOptionsProvider::with_fetcher(
            config(),
            fetcher,
            Arc::new(move || *clock_now.lock().unwrap()),
        );

        assert!(provider.fetch_storage_options().await.is_ok());
        *now.lock().unwrap() += chrono::Duration::seconds(61);

        assert!(provider.fetch_storage_options().await.is_err());
    }

    #[tokio::test]
    async fn rejects_credential_that_expires_during_refresh() {
        let now = Arc::new(Mutex::new(Utc::now()));
        let initial_now = *now.lock().unwrap();
        let fetcher = Arc::new(ClockAdvancingFetcher {
            responses: Mutex::new(
                vec![(
                    chrono::Duration::seconds(2),
                    Ok(AzureSasCredential {
                        token: "sv=1&sig=stale".to_string(),
                        expires_at: initial_now + chrono::Duration::seconds(1),
                    }),
                )]
                .into(),
            ),
            clock: now.clone(),
        });
        let clock_now = now.clone();
        let provider = AzureSasStorageOptionsProvider::with_fetcher(
            config(),
            fetcher,
            Arc::new(move || *clock_now.lock().unwrap()),
        );

        assert!(provider.fetch_storage_options().await.is_err());
    }

    #[test]
    fn provider_id_is_hashed_and_contains_no_credential_material() {
        let id = config().provider_id();
        assert!(id.starts_with("azure_sas_broker_"));
        assert!(!id.contains("client"));
        assert!(!id.contains("tenant"));
    }
}
