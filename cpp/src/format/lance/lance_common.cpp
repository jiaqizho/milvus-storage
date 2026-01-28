// Copyright 2023 Zilliz
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

#include "milvus-storage/format/lance/lance_common.h"

#include <fmt/format.h>

namespace milvus_storage::lance {

static const std::string kLanceUriDelimiter = "?fragment_id=";

arrow::Result<std::pair<std::string, uint64_t>> ParseLanceUri(const std::string& uri) {
  auto pos = uri.find(kLanceUriDelimiter);
  if (pos == std::string::npos) {
    return arrow::Status::Invalid("Invalid uri format: ", uri,
                                  ". Expected format: {base_path}?fragment_id={fragment_id}");
  }

  uint64_t fragment_id = 0;
  try {
    fragment_id = std::stoull(uri.substr(pos + kLanceUriDelimiter.length()));
  } catch (const std::exception& e) {
    return arrow::Status::Invalid(fmt::format("Invalid fragment_id in uri: {}", uri));
  }

  auto base_path = uri.substr(0, pos);
  return std::make_pair(base_path, fragment_id);
}

std::string MakeLanceUri(const std::string& base_path, uint64_t fragment_id) {
  return base_path + kLanceUriDelimiter + std::to_string(fragment_id);
}

arrow::Result<std::string> BuildLanceBaseUri(const ArrowFileSystemConfig& config, const std::string& relative_path) {
  if (config.storage_type == "local") {
    // For local storage, construct absolute path
    return config.root_path + "/" + relative_path;
  }

  const auto& provider = config.cloud_provider;
  const auto& bucket = config.bucket_name;

  if (bucket.empty()) {
    return arrow::Status::Invalid("Bucket name is required for cloud storage");
  }

  std::string scheme;
  if (provider == kCloudProviderAWS) {
    scheme = "s3";
  } else if (provider == kCloudProviderAzure) {
    scheme = "az";
  } else if (provider == kCloudProviderGCP) {
    scheme = "gs";
  } else if (provider == kCloudProviderAliyun) {
    scheme = "oss";
  } else if (provider == kCloudProviderTencent || provider == kCloudProviderHuawei) {
    return arrow::Status::Invalid("Lance does not support cloud provider: " + provider);
  } else {
    return arrow::Status::Invalid("Unknown cloud provider: " + provider);
  }

  return scheme + "://" + bucket + "/" + relative_path;
}

LanceStorageOptions ToLanceStorageOptions(const ArrowFileSystemConfig& config) {
  LanceStorageOptions options;

  // only remote need use `LanceStorageOptions`
  if (config.storage_type == "local") {
    return options;
  }

  const auto& provider = config.cloud_provider;

  // Helper to build endpoint URL with https scheme (default)
  // Returns the endpoint URL and whether HTTP is used
  auto build_endpoint_url = [](const std::string& address) -> std::pair<std::string, bool> {
    if (address.empty()) {
      return {"", false};
    }
    // If already has scheme, check if it's http
    if (address.find("://") != std::string::npos) {
      bool is_http = address.find("http://") == 0;
      return {address, is_http};
    }
    // Default to HTTPS for cloud storage
    return {"https://" + address, false};
  };

  if (provider == kCloudProviderAWS) {
    if (!config.access_key_id.empty()) {
      options["aws_access_key_id"] = config.access_key_id;
    }
    if (!config.access_key_value.empty()) {
      options["aws_secret_access_key"] = config.access_key_value;
    }
    if (!config.region.empty()) {
      options["aws_region"] = config.region;
    }
    if (!config.address.empty()) {
      auto [endpoint, allow_http] = build_endpoint_url(config.address);
      options["aws_endpoint"] = endpoint;
      if (allow_http) {
        options["allow_http"] = "true";
      }
    }

  } else if (provider == kCloudProviderAzure) {
    if (!config.access_key_id.empty()) {
      options["azure_storage_account_name"] = config.access_key_id;
    }
    if (!config.access_key_value.empty()) {
      options["azure_storage_account_key"] = config.access_key_value;
    }
    if (!config.address.empty()) {
      // Azure endpoint requires storage account name in the URL
      // Format: https://<storage_account>.blob.core.windows.net
      // If address doesn't contain account name, prepend it automatically
      std::string address = config.address;
      std::string scheme_prefix;

      // Extract scheme if present
      size_t scheme_pos = address.find("://");
      if (scheme_pos != std::string::npos) {
        scheme_prefix = address.substr(0, scheme_pos + 3);
        address = address.substr(scheme_pos + 3);
      }

      // Prepend account name if not already present
      const std::string& account_name = config.access_key_id;
      if (!account_name.empty() && address.find(account_name + ".") != 0) {
        address = account_name + "." + address;
      }

      // Rebuild full address
      std::string full_address = scheme_prefix + address;

      auto [endpoint, allow_http] = build_endpoint_url(full_address);
      options["azure_endpoint"] = endpoint;
      if (allow_http) {
        options["allow_http"] = "true";
      }
    }

  } else if (provider == kCloudProviderGCP) {
    // GCP uses default credentials, no additional options needed

  } else if (provider == kCloudProviderAliyun) {
    // Aliyun OSS uses opendal with oss_* parameters
    if (!config.access_key_id.empty()) {
      options["oss_access_key_id"] = config.access_key_id;
    }
    if (!config.access_key_value.empty()) {
      options["oss_secret_access_key"] = config.access_key_value;
    }
    if (!config.region.empty()) {
      options["oss_region"] = config.region;
    }
    if (!config.address.empty()) {
      auto [endpoint, allow_http] = build_endpoint_url(config.address);
      options["oss_endpoint"] = endpoint;
      if (allow_http) {
        options["allow_http"] = "true";
      }
    }

  } else if (provider == kCloudProviderTencent || provider == kCloudProviderHuawei) {
    throw LanceException("Lance does not support cloud provider: " + provider);

  } else {
    throw LanceException("Unknown cloud provider: " + provider);
  }

  return options;
}

}  // namespace milvus_storage::lance
