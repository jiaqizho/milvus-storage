// Copyright 2024 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <aws/core/Aws.h>
#include <aws/core/utils/logging/LogLevel.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/curl/CurlHttpClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <curl/curl.h>

#include "milvus-storage/filesystem/s3/s3_client.h"
#include "milvus-storage/filesystem/s3/s3_global.h"
#include "milvus-storage/filesystem/s3/s3_filesystem_producer.h"
#include "milvus-storage/common/arrow_util.h"

#include "test_env.h"

namespace milvus_storage::test {

// Returns true only when the environment targets a real cloud endpoint with HTTPS.
// Skips local filesystem, MinIO (http://...), and other non-TLS setups.
static bool IsTlsCloudEnv() {
  if (!IsCloudEnv()) {
    return false;
  }
  auto address = GetEnvVar(ENV_VAR_ADDRESS).ValueOr("");
  // MinIO and local S3-compatible services use "http://..." — no TLS.
  // Real cloud endpoints (e.g. "s3.us-west-2.amazonaws.com") have no scheme prefix.
  if (address.empty() || address.rfind("http://", 0) == 0) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Custom AWS SDK logger that captures log messages for TLS version inspection.
// ---------------------------------------------------------------------------
class CapturingLogger : public Aws::Utils::Logging::LogSystemInterface {
  public:
  Aws::Utils::Logging::LogLevel GetLogLevel() const override { return Aws::Utils::Logging::LogLevel::Debug; }

  void Log(Aws::Utils::Logging::LogLevel logLevel, const char* tag, const char* formatStr, ...) override {
    char buf[4096];
    va_list args;
    va_start(args, formatStr);
    vsnprintf(buf, sizeof(buf), formatStr, args);
    va_end(args);

    std::lock_guard<std::mutex> lock(mutex_);
    messages_.emplace_back(buf);
  }

  void vaLog(Aws::Utils::Logging::LogLevel logLevel, const char* tag, const char* formatStr, va_list args) override {
    char buf[4096];
    vsnprintf(buf, sizeof(buf), formatStr, args);

    std::lock_guard<std::mutex> lock(mutex_);
    messages_.emplace_back(buf);
  }

  void LogStream(Aws::Utils::Logging::LogLevel logLevel,
                 const char* tag,
                 const Aws::OStringStream& messageStream) override {
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.emplace_back(messageStream.str());
  }

  void Flush() override {}

  // Search captured logs for TLS version string.
  // Different TLS backends produce different curl verbose formats:
  //   OpenSSL:          "SSL connection using TLSv1.3 / TLS_AES_128_GCM_SHA256"
  //   Secure Transport: "TLS 1.2 connection using TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
  std::string FindTlsVersion() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::regex openssl_regex(R"(SSL connection using (TLSv[\d.]+))");
    std::regex sectransp_regex(R"((TLS [\d.]+) connection using)");
    for (const auto& msg : messages_) {
      std::smatch match;
      if (std::regex_search(msg, match, openssl_regex)) {
        return match[1].str();
      }
      if (std::regex_search(msg, match, sectransp_regex)) {
        std::string ver = match[1].str();
        return "TLSv" + ver.substr(4);  // "TLS 1.2" -> "TLSv1.2"
      }
    }
    return "";
  }

  std::vector<std::string> FilterMessages(const std::string& keyword) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    for (const auto& msg : messages_) {
      if (msg.find(keyword) != std::string::npos) {
        result.push_back(msg);
      }
    }
    return result;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return messages_.size();
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.clear();
  }

  private:
  mutable std::mutex mutex_;
  std::vector<std::string> messages_;
};

// ---------------------------------------------------------------------------
// TLS 1.3 enforcing CurlHttpClient subclass.
// Uses the AWS SDK's OverrideOptionsOnConnectionHandle() virtual hook
// to override CURLOPT_SSLVERSION AFTER the default configuration.
// ---------------------------------------------------------------------------
class TLS13CurlHttpClient : public Aws::Http::CurlHttpClient {
  public:
  using CurlHttpClient::CurlHttpClient;

  protected:
  void OverrideOptionsOnConnectionHandle(CURL* handle) const override {
    curl_easy_setopt(handle, CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_3);
  }
};

// ---------------------------------------------------------------------------
// Custom HttpClientFactory that creates TLS13CurlHttpClient instances.
// ---------------------------------------------------------------------------
class TLS13HttpClientFactory : public Aws::Http::HttpClientFactory {
  public:
  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
      const Aws::Client::ClientConfiguration& config) const override {
    return Aws::MakeShared<TLS13CurlHttpClient>("TLS13Factory", config);
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::String& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& streamFactory) const override {
    return CreateHttpRequest(Aws::Http::URI(uri), method, streamFactory);
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(const Aws::Http::URI& uri,
                                                            Aws::Http::HttpMethod method,
                                                            const Aws::IOStreamFactory& streamFactory) const override {
    auto request = Aws::MakeShared<Aws::Http::Standard::StandardHttpRequest>("TLS13Factory", uri, method);
    request->SetResponseStreamFactory(streamFactory);
    return request;
  }
};

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class S3TlsVersionTest : public ::testing::Test {
  protected:
  void SetUp() override {
    if (!IsTlsCloudEnv()) {
      GTEST_SKIP() << "Skipping: requires a cloud endpoint with HTTPS "
                   << "(not local or MinIO over HTTP)";
    }

    bucket_ = GetEnvVar(ENV_VAR_BUCKET_NAME).ValueOr("test-bucket");

    api::Properties properties;
    ASSERT_STATUS_OK(InitTestProperties(properties));
    api::SetValue(properties, PROPERTY_FS_USE_SSL, "true");
    ASSERT_AND_ASSIGN(fs_config_, GetFileSystemConfig(properties));
  }

  // Build an S3 client with curl verbose tracing and make a PutObject request.
  // Returns the negotiated TLS version string (e.g. "TLSv1.2" or "TLSv1.3").
  std::string ConnectAndGetTlsVersion(const std::shared_ptr<CapturingLogger>& logger) {
    logger->Clear();

    S3FileSystemProducer producer(fs_config_);
    producer.InitS3();
    auto s3_options_result = producer.CreateS3Options();
    EXPECT_TRUE(s3_options_result.ok()) << s3_options_result.status().ToString();
    auto s3_options = std::move(s3_options_result).ValueOrDie();

    ClientBuilder builder(s3_options);
    builder.mutable_config()->enableHttpClientTrace = true;
    auto client_result = builder.BuildClient();
    EXPECT_TRUE(client_result.ok()) << client_result.status().ToString();
    auto client_holder = std::move(client_result).ValueOrDie();

    auto lock_result = client_holder->Lock();
    EXPECT_TRUE(lock_result.ok()) << lock_result.status().ToString();
    auto client_lock = std::move(lock_result).ValueOrDie();

    Aws::S3::Model::PutObjectRequest put_request;
    put_request.SetBucket(bucket_.c_str());
    put_request.SetKey("unittest/tls_version_test.txt");
    auto body = Aws::MakeShared<Aws::StringStream>("TlsTest");
    (*body) << "tls version test";
    put_request.SetBody(body);

    auto outcome = client_lock.Move()->PutObject(put_request);
    EXPECT_TRUE(outcome.IsSuccess()) << "PutObject failed: " << outcome.GetError().GetMessage();

    return logger->FindTlsVersion();
  }

  std::string bucket_;
  ArrowFileSystemConfig fs_config_;
};

// ---------------------------------------------------------------------------
// Test: Verify default TLS version (should be >= TLS 1.2)
// ---------------------------------------------------------------------------
TEST_F(S3TlsVersionTest, VerifyDefaultTlsVersion) {
  S3FileSystemProducer producer(fs_config_);
  producer.InitS3();

  auto logger = Aws::MakeShared<CapturingLogger>("TlsTest");
  Aws::Utils::Logging::InitializeAWSLogging(logger);

  std::string tls_version = ConnectAndGetTlsVersion(logger);

  if (tls_version.empty()) {
    auto tls_msgs = logger->FilterMessages("TLS");
    auto ssl_msgs = logger->FilterMessages("SSL");
    std::cerr << "TLS msgs: " << tls_msgs.size() << ", SSL msgs: " << ssl_msgs.size() << std::endl;
    for (const auto& msg : tls_msgs) std::cerr << "  " << msg << std::endl;
    for (const auto& msg : ssl_msgs) std::cerr << "  " << msg << std::endl;
    FAIL() << "Could not find TLS version in curl verbose output.";
  }

  std::cout << ">>> Default negotiated TLS version: " << tls_version << std::endl;

  EXPECT_TRUE(tls_version == "TLSv1.2" || tls_version == "TLSv1.3") << "Unexpected TLS version: " << tls_version;
}

// ---------------------------------------------------------------------------
// Test: Enforce TLS 1.3 via custom HttpClientFactory
// ---------------------------------------------------------------------------
TEST_F(S3TlsVersionTest, EnforceTls13ViaCustomFactory) {
  S3FileSystemProducer producer(fs_config_);
  producer.InitS3();

  // Replace the global HTTP client factory with our TLS 1.3 version.
  auto tls13_factory = Aws::MakeShared<TLS13HttpClientFactory>("TLS13Test");
  Aws::Http::SetHttpClientFactory(tls13_factory);

  auto logger = Aws::MakeShared<CapturingLogger>("TlsTest");
  Aws::Utils::Logging::InitializeAWSLogging(logger);

  std::string tls_version = ConnectAndGetTlsVersion(logger);

  if (tls_version.empty()) {
    auto tls_msgs = logger->FilterMessages("TLS");
    auto ssl_msgs = logger->FilterMessages("SSL");
    std::cerr << "TLS msgs: " << tls_msgs.size() << ", SSL msgs: " << ssl_msgs.size() << std::endl;
    for (const auto& msg : tls_msgs) std::cerr << "  " << msg << std::endl;
    for (const auto& msg : ssl_msgs) std::cerr << "  " << msg << std::endl;
    FAIL() << "Could not find TLS version after enforcing TLS 1.3. "
           << "Note: Secure Transport (macOS) does not support CURL_SSLVERSION_TLSv1_3. "
           << "This test requires Linux with OpenSSL backend.";
  }

  std::cout << ">>> Enforced TLS version: " << tls_version << std::endl;

  EXPECT_EQ(tls_version, "TLSv1.3") << "Expected TLSv1.3 but got: " << tls_version;
}

}  // namespace milvus_storage::test
