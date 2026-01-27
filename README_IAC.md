# Azure ML Workspace Terraform Setup

This repository contains Terraform configuration to provision an Azure Machine Learning (AML) workspace and its core dependencies, including compute resources.

## What this code creates

- **Resource group**
  - Name generated from a random pet name + `-rg` suffix.
- **Supporting resources for AML**
  - **Application Insights** for monitoring.
  - **Key Vault** for secrets.
  - **Storage Account** as the default AML datastore.
  - **Azure Container Registry (ACR)** for model images.
- **Azure Machine Learning workspace**
  - Wired to Application Insights, Key Vault, Storage Account, and ACR.
  - System-assigned managed identity.
- **Compute resources**
  - **Compute instance** for interactive work (e.g., notebooks).
  - **Compute cluster** for training jobs with autoscaling.
- **Randomized naming**
  - `random_pet`, `random_integer`, and `random_string` are used to generate unique resource names and avoid global-name collisions.

## File overview

- **`providers.tf`**
  - Declares Terraform version and required providers (`azurerm`, `random`).
  - Configures the `azurerm` provider using Azure service principal credentials and subscription info.
- **`main.tf`**
  - Fetches current Azure client configuration.
  - Creates a resource group.
  - Creates random name components (pet prefix and integer suffix).
- **`workspace.tf`**
  - Creates Application Insights, Key Vault, Storage Account, and Container Registry.
  - Creates the Azure Machine Learning workspace that depends on those resources.
- **`compute.tf`**
  - Generates a random string for a unique compute instance name.
  - Creates an AML compute instance.
  - Creates an AML compute cluster with autoscaling.
- **`variables.tf`**
  - Defines input variables such as `environment`, `location`, and `prefix`.
- **`outputs.tf`**
  - Exposes the names of the Key Vault, Storage Account, Container Registry, AML workspace, compute instance, and compute cluster.

> Note: Some sensitive-related configuration (like keys or credentials) may be defined in files intentionally ignored by Git (e.g., `key.tf`). Make sure to provide those values securely, for example with a `terraform.tfvars` file that you do not commit.

## Prerequisites

- **Tools**
  - Terraform CLI `>= 1.0`.
- **Azure**
  - An Azure subscription.
  - A service principal with permissions to create resource groups and AML resources.

## Required inputs

From the visible code, these variables are used:

- **`subscription_id`** – Azure subscription to deploy into (used in `provider "azurerm"`).
- **`client_id`** – Service principal application (client) ID.
- **`client_secret`** – Service principal client secret.
- **`tenant_id`** – Azure AD tenant ID.
- **`environment`** – Environment name (e.g., `dev`, `qa`, `prod`). Default: `dev`.
- **`location`** – Azure region, e.g., `eastus`. Default: `eastus`.
- **`prefix`** – Short prefix used in resource names. Default: `ml`.

The credential-related variables are likely declared in an ignored file (for example, `key.tf`) or passed via `terraform.tfvars` or environment variables.

### Example `terraform.tfvars`

```hcl
subscription_id = "00000000-0000-0000-0000-000000000000"
client_id       = "11111111-1111-1111-1111-111111111111"
client_secret   = "your-sp-secret"
tenant_id       = "22222222-2222-2222-2222-222222222222"

# Optional overrides
environment = "dev"
location    = "eastus"
prefix      = "ml"
```

Do **not** commit this file to source control.

## How to use

1. **Initialize Terraform**

   ```bash
   terraform init
   ```

2. **Review the plan**

   ```bash
   terraform plan
   ```

3. **Apply the configuration**

   ```bash
   terraform apply -auto-approve
   ```

   Confirm with `yes` when prompted. After completion, Terraform will print the outputs defined in `outputs.tf`.

4. **Destroy the resources (when no longer needed)**

   ```bash
   terraform destroy
   ```

   This removes the created resource group and all the associated resources.

- terraform state list
- terraform destroy --target name_of_resource

## Outputs

After a successful apply, you should see values for:

- `key_vault_name`
- `storage_account_name`
- `container_registry_name`
- `machine_learning_workspace_name`
- `machine_learning_compute_instance_name`
- `machine_learning_compute_cluster_name`

Use these in scripts, tooling, or the Azure Portal to quickly find and work with the deployed resources.

## tfsec

01. Install tfsec on Windows

```bash
choco install tfsec
```

```bash
tfsec .
```

