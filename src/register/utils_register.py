import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
import os


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