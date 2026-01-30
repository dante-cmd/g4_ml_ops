import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
import os
import time
from datetime import datetime
from azure.core.credentials import AccessToken

class AzureMLV1Credential:
    """Wrapper to use Azure ML V1 Authentication object as a V2 TokenCredential"""
    def __init__(self, v1_auth):
        self.v1_auth = v1_auth

    def get_token(self, *scopes, **kwargs):
        # Map scope to resource (simple heuristic)
        resource = "https://management.azure.com/"
        for scope in scopes:
            if "management.azure.com" in scope:
                resource = "https://management.azure.com/"
            elif "ml.azure.com" in scope:
                resource = "https://ml.azure.com/"
        
        # Get token from V1 auth object
        token_obj = self.v1_auth.get_token(resource)
        
        # Calculate expiry
        # 'expiresOn' is typically ISO8601 string
        expires_on = int(time.time() + 3600) # Default fallback
        if 'expiresOn' in token_obj:
            try:
                # Remove 'Z' if present for fromisoformat compatibility in older python
                ts_str = token_obj['expiresOn'].replace('Z', '+00:00')
                dt = datetime.fromisoformat(ts_str)
                expires_on = int(dt.timestamp())
            except:
                pass
                
        return AccessToken(token_obj['accessToken'], expires_on)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input_path", type=str, help="Ruta donde está el modelo entrenado")
    parser.add_argument("--model_name", type=str, help="Nombre para registrar el modelo")
    args = parser.parse_args()
    return args


def get_ml_client():
    # Initialize ML Client
    try:
        credential = DefaultAzureCredential()
        
        # First, try to use the Azure ML Run context (when running inside Azure ML)
        try:
            from azureml.core import Run
            run = Run.get_context()
            
            # Check if we're running in Azure ML (not offline run)
            if hasattr(run, 'experiment'):
                print("Running in Azure ML. Using workspace from Run context...")
                workspace = run.experiment.workspace
                
                # Use the V1 workspace authentication to create a V2-compatible credential
                # This fixes 'DefaultAzureCredential failed' errors in AML Jobs
                v1_auth = workspace.service_context.authentication
                credential = AzureMLV1Credential(v1_auth)
                
                ml_client = MLClient(
                    credential=credential,
                    subscription_id=workspace.subscription_id,
                    resource_group_name=workspace.resource_group,
                    workspace_name=workspace.name
                )
                return ml_client
        except Exception as run_context_error:
            print(f"Could not use Run context: {run_context_error}")
        
        # Check for environment variables for explicit configuration (Actions Support)
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
        resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
        workspace_name = os.environ.get("AZURE_WORKSPACE_NAME")

        if subscription_id and resource_group and workspace_name:
            print("Initializing MLClient using environment variables...")
            ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
        else:
            print("Environment variables missing. Attempting to use config.json...")
            ml_client = MLClient.from_config(credential=credential)
    except Exception as e:
        print(f"Error initializing MLClient: {e}")
        raise
    return ml_client