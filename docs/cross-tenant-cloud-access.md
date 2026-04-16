# Cross-Tenant Cloud Storage Access Research

## Background

In the External Table scenario, our service needs to access cloud storage resources in a customer's account. All three major cloud providers (AWS / GCP / Azure) offer cross-tenant temporary credential mechanisms that follow the same core pattern:

```
Our Instance (Account A)
    |
    +-- 1. Use its own identity credential (bound to the instance, auto-obtained, no key management)
    |
    +-- 2. Request the cloud provider's token service:
    |      "I need to access Account B's resources, give me temporary credentials"
    |
    +-- 3. Cloud provider verifies: has Account B pre-authorized Account A's identity?
    |
    +-- 4. Verification passes -> returns temporary credentials -> access Account B's storage
```

Prerequisite: **The customer (Account B) must pre-configure the authorization.** Our side does not need to hold any of the customer's keys.

---

## 1. Core Concepts Comparison

### 1.1 Account / Tenant Hierarchy

| Concept | AWS | GCP | Azure |
|---------|-----|-----|-------|
| Top-level isolation unit | **Account** (12-digit ID) | **Organization** | **Tenant** (= Entra ID Directory) |
| Resource ownership unit | Account | **Project** (similar to sub-account) | **Subscription** |
| Cross-org access | Cross-Account | Cross-Project / Cross-Organization | Cross-Tenant |

### 1.2 Identities

| Concept | AWS | GCP | Azure |
|---------|-----|-----|-------|
| Human user | IAM User | Google Account | Entra ID User |
| Programmatic / service identity | IAM Role (assumed by EC2, etc.) | **Service Account** (SA) | **Service Principal** (SP) |
| Machine-bound identity | EC2 Instance Profile | SA attached to VM | **Managed Identity** (MI) |

**GCP Service Account (SA)**:
- An email-formatted identity: `my-sa@my-project.iam.gserviceaccount.com`
- Acts as both an identity (can authenticate) and a resource (can be managed / impersonated by others)
- Can generate a JSON Key file (long-lived credential), or obtain temporary tokens via VM metadata
- Key difference from AWS: AWS Roles are "assumed"; GCP SAs are "impersonated"

**Azure Service Principal (SP)**:
- An instance of an App Registration within a specific Tenant
- App Registration is the application definition (globally unique); SP is its projection in a given Tenant
- Analogy: App Registration = class definition, SP = object instance

**Azure Managed Identity (MI)**:
- An Azure-managed SP, bound to a specific resource (VM, Function, etc.)
- No manual key management required; Azure handles rotation automatically
- Equivalent to AWS EC2 Instance Profile + GCP VM-attached SA

### 1.3 Credential Types

| Credential | AWS | GCP | Azure |
|------------|-----|-----|-------|
| Long-lived key | Access Key + Secret Key | SA JSON Key file | Client Secret / Certificate |
| Temporary credential | STS Token (AssumeRole) | OAuth2 Access Token | OAuth2 Bearer Token |
| Machine-auto credential | EC2 Metadata (169.254.169.254) | VM Metadata (metadata.google.internal) | IMDS (169.254.169.254) |

### 1.4 Permission Models

| Concept | AWS | GCP | Azure |
|---------|-----|-----|-------|
| Permission mechanism | IAM Policy (JSON) | IAM Binding (Role -> Member) | RBAC (Role Assignment) |
| Storage read/write permissions | `s3:GetObject`, etc. | `roles/storage.objectViewer`, etc. | `Storage Blob Data Reader`, etc. |

---

## 2. Cross-Tenant Access Flows

### 2.1 Official Terminology

| Provider | Mechanism Name | Core API / Protocol |
|----------|---------------|---------------------|
| **AWS** | **STS AssumeRole** | `sts:AssumeRole` |
| **GCP** | **Service Account Impersonation** | IAM Credentials API `generateAccessToken` |
| **Azure** | **Multi-Tenant App + Federated Identity Credential** | OAuth2 Client Credentials with MI-based client assertion |

### 2.2 What the Customer Gives Us vs. What They Pre-Configure

| Provider | What customer gives us | What customer pre-configures (transparent to us) |
|----------|----------------------|--------------------------------------------------|
| **AWS** | `role_arn` | Create IAM Role + Trust Policy allowing our Account |
| **GCP** | `target_sa_email` | Create SA + grant our SA the `serviceAccountTokenCreator` role |
| **Azure** | `client_id` + `tenant_id` | Admin Consent + RBAC + Federated Credential trusting our MI |

From our code's perspective, all three are the same: **receive identifier strings from customer -> pass to SDK -> SDK auto-exchanges for temporary credentials -> access resources**. The authorization setup is entirely on the customer's side.

### 2.3 AWS STS AssumeRole (Already Implemented - Reference)

```
Account A (ours)                    Account B (customer)
+----------------+                  +--------------------+
| EC2 Instance   |                  | IAM Role           |
| (has Instance  |-- AssumeRole --> | arn:aws:iam::      |
|  Profile)      |   w/ role_arn    | 123456:role/X      |
|                |<- temp AKSK     |                     |
|                |   + Session Tok  | Trust Policy:       |
|                |-- use temp   --> | allows Account A's  |
|                |   creds to S3    | identity to assume  |
+----------------+                  +--------------------+
```

Parameters: `role_arn`, `session_name`, `external_id`

### 2.4 GCP Service Account Impersonation

```
Project A (ours)                    Project B (customer)
+----------------+                  +-------------------------------+
| VM with SA-A   |                  | Service Account SA-B          |
| (sa-a@         |-- generateAccess | (sa-b@project-b.iam.          |
|  proj-a.iam.)  |   Token(SA-B)-->|  gserviceaccount.com)          |
|                |<- OAuth2 Token  |                                |
|                |                  | IAM Binding:                   |
|                |-- use token  -->| SA-A has                       |
|                |   to access GCS  | serviceAccountTokenCreator    |
|                |                  | role on SA-B                   |
+----------------+                  +-------------------------------+
```

Key differences from AWS:
- AWS passes a **Role ARN** (resource identifier); GCP passes a **SA Email** (target SA's email address)
- AWS uses Trust Policy to control who can assume; GCP uses IAM Binding to control who can impersonate

### 2.5 Azure Cross-Tenant via Managed Identity + Federated Credential

```
Tenant A (ours)                     Tenant B (customer)
+----------------+                  +-----------------------+
| Node with      |                  | Multi-Tenant App      |
| Managed        |                  | (client_id)           |
| Identity       |                  |                       |
|                |  1. Get MI token |                       |
|                |     from IMDS    | Federated Credential: |
|                |     (automatic)  | trusts our MI         |
|                |                  |                       |
|                |  2. POST to:     |                       |
|                |  login.microsoft |                       |
|                |  online.com/     |                       |
|                |  {tenant_id}/    |                       |
|                |  oauth2/v2.0/    |                       |
|                |  token           |                       |
|                |-- MI token as -->| 3. Verify MI token    |
|                |   assertion      |    Issue Bearer Token |
|                |<- Bearer Token --|                       |
|                |                  | RBAC:                 |
|                |-- use token  --> | Storage Blob Data     |
|                |   to access Blob | Contributor           |
|                |                  |                       |
|                |                  | Storage Account:      |
|                |                  | (account_name)        |
+----------------+                  +-----------------------+
```

Key characteristics:
- **No secrets involved**: the Node's Managed Identity provides the base credential automatically via IMDS, similar to AWS Instance Profile
- Customer gives us two identifiers: `client_id` (which App to authenticate as) + `tenant_id` (which Tenant to target)
- Additionally, `account_name` identifies which storage account to access

---

## 3. Side-by-Side Summary

| Step | AWS | GCP | Azure |
|------|-----|-----|-------|
| Identity bound to instance | Instance Profile (IAM Role) | Attached Service Account | Managed Identity |
| Token service | STS (`AssumeRole`) | IAM Credentials API (`generateAccessToken`) | Entra ID OAuth endpoint |
| Customer-provided identifiers | `role_arn` | target SA email | `client_id` + `tenant_id` |
| How customer pre-authorizes | Trust Policy allowing our Account | IAM Binding granting our SA `TokenCreator` role | Admin Consent + RBAC + Federated Credential trusting our MI |
| Returned temporary credential | Temp AK/SK + Session Token | OAuth2 Access Token | OAuth2 Bearer Token |
| Secrets management | None (Instance Profile) | None (VM metadata) | None (Managed Identity) |

---

## 4. Rust Library Support

### 4.1 opendal (used by Iceberg)

| Feature | Support | Notes |
|---------|---------|-------|
| AWS AssumeRole | Supported | `client.assume-role.arn` and related config keys |
| GCP SA Impersonation | **Supported** | reqsign auto-parses `impersonated_service_account` and `external_account` credential JSON types |
| Azure Cross-Tenant | **Supported** | `tenant_id`, `client_id` config keys; MI provides base credential |

GCP usage: pass a correctly-formatted credential JSON via `credential` / `credential_path` to opendal. reqsign handles the impersonation flow automatically -- no new explicit parameters needed.

Supported credential JSON types:
- `type: "impersonated_service_account"` -- contains `service_account_impersonation_url` and source credentials
- `type: "external_account"` with `service_account_impersonation_url` -- Workload Identity Federation

### 4.2 object_store (used by Lance)

| Feature | Support | Notes |
|---------|---------|-------|
| AWS AssumeRole | Supported | `aws_role_arn` and related config keys |
| GCP SA Impersonation | **Not supported** | Only recognizes `service_account` and `authorized_user`; requires custom CredentialProvider. Upstream issue: [apache/arrow-rs-object-store#258](https://github.com/apache/arrow-rs-object-store/issues/258) |
| Azure Cross-Tenant | **Supported** | `tenant_id`, `client_id` config keys; MI provides base credential |

---

## 5. Existing AWS ARN Implementation in Codebase

### 5.1 Configuration Flow

```
External Table Config (user)
    |
    v
properties: fs.role_arn, fs.session_name, fs.external_id, fs.load_frequency
(external tables: extfs.<alias>.role_arn, extfs.<alias>.session_name ...)
    |
    v
ArrowFileSystemConfig::create_file_system_config()  [fs.cpp]
    |
    v
ArrowFileSystemConfig { role_arn, session_name, external_id, load_frequency }
    |
    v
Format-specific conversion:
    +-- Lance: ToStorageOptions()  [lance_common.cpp]
    |   +-- aws_role_arn, aws_session_name, aws_external_id
    |
    +-- Iceberg: ToStorageOptions()  [iceberg_common.cpp]
        +-- client.assume-role.arn, client.assume-role.session-name, client.assume-role.external-id
    |
    v
Rust FFI bridge -> Lance/opendal handles STS AssumeRole
```

### 5.2 Key Mapping Between Lance and Iceberg

| Parameter | Lance (object_store) | Iceberg (opendal) |
|-----------|---------------------|-------------------|
| Role ARN | `aws_role_arn` | `client.assume-role.arn` |
| Session Name | `aws_session_name` | `client.assume-role.session-name` |
| External ID | `aws_external_id` | `client.assume-role.external-id` |
| Credential Refresh | `aws_credential_refresh_secs` | (handled internally by opendal) |

---

## 6. Implementation Plan

### 6.1 Azure Cross-Tenant (Both Libraries Natively Support)

Add `tenant_id`, `client_id` fields to `ArrowFileSystemConfig`, then:

- **Lance** (`lance_common.cpp`): set `azure_tenant_id`, `azure_client_id`; no `client_secret` needed since MI provides the base credential
- **Iceberg** (`iceberg_common.cpp`): set `adls.tenant-id`, `adls.client-id`; MI provides the base credential

Follows the same pattern as AWS ARN -- when cross-tenant config is detected, set the corresponding keys and skip account key.

Parameters from customer: `account_name`, `client_id`, `tenant_id`

### 6.2 GCP Cross-Tenant

- **Iceberg (opendal)**: pass a correctly-formatted credential JSON to opendal (via `credential` or `credential_path`). reqsign handles impersonation automatically. No new explicit parameters needed.
- **Lance (object_store)**: object_store **does not support** the `impersonated_service_account` credential JSON type. Options:
  1. Implement a custom `GcpCredentialProvider` that manually calls the `generateAccessToken` API
  2. Or rely on the runtime environment (GCE VM) whose SA already has cross-project IAM permissions (no impersonation needed)

Parameters from customer: `target_sa_email`
