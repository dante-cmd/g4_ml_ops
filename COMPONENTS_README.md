# ML Pipeline Components and CI/CD Integration

This repository contains modular ML pipeline components designed for seamless CI/CD integration with Azure Machine Learning.

## Project Structure

```
components/                 # Reusable ML components
├── etl/                    # Data extraction, transformation, loading
│   ├── etl_component.py    # ETL component definition
│   └── spec.json           # Component specification
├── features/               # Feature engineering components
├── models/                 # Model training components
└── prediction/             # Prediction components
pipelines/                  # Pipeline definitions
├── base/                   # Basic pipeline configuration
│   └── base_pipeline.py    # Base pipeline definition
├── better/                 # Enhanced pipeline configuration
├── production/             # Production-ready pipeline
└── run_pipeline.py         # Main pipeline execution script
.github/
└── workflows/
    └── ml_pipeline.yml     # CI/CD workflow definition
```

## CI/CD Integration

### GitHub Actions Workflow

The `.github/workflows/ml_pipeline.yml` file defines the CI/CD workflow that:
1. Checks out the code on push/PR to main/develop branches
2. Sets up Python environment
3. Authenticates with Azure
4. Runs the ML pipeline
5. Uploads outputs as artifacts

### Required Secrets

Set up the following secrets in your GitHub repository:

- `AZURE_CREDENTIALS`: Azure service principal credentials in JSON format
- `AZURE_SUBSCRIPTION_ID`: Azure subscription ID
- `RESOURCE_GROUP`: Azure resource group name
- `WORKSPACE_NAME`: Azure ML workspace name
- `AML_ENVIRONMENT`: Azure ML environment name and version
- `AML_COMPUTE_TARGET`: Azure ML compute cluster name
- `PLATINUM_VERSION`: Platinum dataset version
- `FEATS_VERSION`: Features dataset version
- `TARGET_VERSION`: Target dataset version
- `AML_EXPERIMENT_NAME`: Azure ML experiment name

### Setting up Azure Credentials

To generate the Azure credentials for GitHub Actions:

```bash
az login
az ad sp create-for-rbac --name "github-actions-sp" --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/{resource-group} \
  --sdk-auth
```

### Local Development

To run the pipeline locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python pipelines/run_pipeline.py
```

### Environment Variables

The pipeline supports the following environment variables:

- `AML_ENVIRONMENT`: Override the default environment (default: 'customized-env:1')
- `AML_COMPUTE_TARGET`: Override the default compute target (default: 'cibnew')
- `PLATINUM_VERSION`: Platinum dataset version (default: 'v1')
- `FEATS_VERSION`: Features dataset version (default: 'v1')
- `TARGET_VERSION`: Target dataset version (default: 'v1')
- `AML_EXPERIMENT_NAME`: Experiment name (default: 'pipeline-base-cicd')

## Component Architecture

Each component is defined as a reusable unit that can be:
1. Used in multiple pipelines
2. Tested independently
3. Versioned separately
4. Deployed through CI/CD

The base pipeline demonstrates how to compose components into a complete ML workflow.

## Deployment Process

1. Code changes are pushed to the repository
2. GitHub Actions triggers the workflow on configured branches
3. The workflow authenticates with Azure using stored credentials
4. The pipeline executes in Azure ML
5. Results are logged and artifacts are stored
6. Notifications can be configured for success/failure

## Best Practices

- Keep components small and focused on single responsibilities
- Use semantic versioning for components and pipelines
- Store sensitive information in repository secrets, not in code
- Test components individually before integrating into pipelines
- Monitor pipeline runs and set up alerts for failures