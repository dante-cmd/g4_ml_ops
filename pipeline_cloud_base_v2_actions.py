from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.ai.ml.constants import AssetTypes
import json
import os
import sys

# Add src to path to import version fetcher if needed, or just define helpers
sys.path.append("./src")

def get_mapping_tipos(periodo: int) -> dict:
    mod = periodo % 100
    tipos = {
        "inicial_estacional": False,
        "continuidad_estacional": False,
        "inicial_regular": True,
        "continuidad_regular": True
    }
    if mod == 1:
        tipos["inicial_estacional"] = True
    elif mod == 2:
        tipos["inicial_estacional"] = True
        tipos["continuidad_estacional"] = True
    return tipos

def build_pipeline_base_func(periodos, model_periodo, script_types, compute, environment, versions):
    @dsl.pipeline(description="Pipeline Cloud Base (Replicating PowerShell)")
    def pipeline_base_func(ult_periodo, n_periodos, all_periodos, 
                           platinum_version, feats_version_default, target_version_default,
                           exp_model_name, exp_predict_name):
        
        # ---------------------------- ETL -----------------------------------
        etl_component = command(
            code="./src/data",
            command="python etl.py \
                --input_datastore ${{inputs.input_datastore}} \
                --output_datastore ${{outputs.output_datastore}} \
                --platinum_version ${{inputs.platinum_version}} \
                --ult_periodo ${{inputs.ult_periodo}} \
                --n_periodos ${{inputs.n_periodos}} \
                --all_periodos ${{inputs.all_periodos}}",
            inputs={
                "input_datastore": Input(type=AssetTypes.URI_FOLDER),
                "platinum_version": Input(type="string"),
                "ult_periodo": Input(type="integer"),
                "n_periodos": Input(type="string"),
                "all_periodos": Input(type="string")
            },
            outputs={
                "output_datastore": Output(type=AssetTypes.URI_FOLDER)
            },
            environment=environment,
            display_name="etl_component"
        )

        etl_step = etl_component(
            input_datastore=Input(
                type=AssetTypes.URI_FOLDER, 
                path="azureml://datastores/base_data/paths/rawdata"),
            platinum_version=platinum_version,
            ult_periodo=ult_periodo,
            n_periodos=n_periodos,
            all_periodos=all_periodos
        )
        etl_step.compute = compute
        etl_step.name = "etl_step"
        etl_step.display_name = "ETL"
        
        etl_step.outputs.output_datastore = Output(
            type=AssetTypes.URI_FOLDER, 
            path="azureml://datastores/ml_data/paths/platinumdata",
            mode="rw_mount")

        platinum_data_output = etl_step.outputs.output_datastore

        # ---------------------------- Features & Models (Model Periodo) -------------------------
        # Logic: For model_periodo, run features then models
        
        mapping_tipos_model = get_mapping_tipos(model_periodo)
        
    return pipeline_base_func

if __name__ == '__main__':
    # Configuration matches pipeline_local_base.ps1 parameters roughly
    periodos = [202506, 202507, 202508, 202509, 202510, 202511]
    ult_periodo = max(periodos)
    model_periodo = min(periodos)
    n_periodos = "None"
    all_periodos = "True"
    
    script_types = ["continuidad_regular", "inicial_regular", "continuidad_estacional", "inicial_estacional"]
    
    environment = 'customized-env:1'
    compute = 'cibnew'
    
    # Try to fetch version.json if exists or simulate
    pass_versions = {}
    try:
        # Assuming we can run the fetch script or read the file if it exists locally
        pass 
    except:
        print("Could not fetch versions, using defaults")
    
    # Initialize ML Client
    try:
        credential = DefaultAzureCredential()
        
        # Check for environment variables for explicit configuration (Actions Support)
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
        resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
        workspace_name = os.environ.get("AZURE_WORKSPACE_NAME")
        print("subscription_id",subscription_id)
        print("resource_group",resource_group)
        print("workspace_name",workspace_name)

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
        print(f"Skipping MLClient init: {e}")
        ml_client = None

    pipeline_func = build_pipeline_base_func(
        periodos=periodos,
        model_periodo=model_periodo,
        script_types=script_types,
        compute=compute,
        environment=environment,
        versions=pass_versions
    )

    pipeline_job = pipeline_func(
        ult_periodo=ult_periodo,
        n_periodos=n_periodos,
        all_periodos=all_periodos,
        platinum_version="v1", # Default or from fetch
        feats_version_default="v1",
        target_version_default="v1",
        exp_model_name="Train Models",
        exp_predict_name="Predict Models"
    )

    if ml_client:
        print("Submitting pipeline...")
        submitted_job = ml_client.jobs.create_or_update(
            pipeline_job, experiment_name="pipeline_base_cloud"
        )
        print(f"Pipeline submitted: {submitted_job.studio_url}")
