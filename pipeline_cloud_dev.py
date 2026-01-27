from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.ai.ml.constants import AssetTypes
import numpy as np

def get_mapping_tipos(periodo: int) -> dict:
    if periodo % 100 == 1:
        return {
            'inicial_estacional': True,
            'continuidad_estacional': False,
            'inicial_regular': True,
            'continuidad_regular': True
        }
    elif periodo % 100 == 2:
        return {
            'inicial_estacional': True,
            'continuidad_estacional': True,
            'inicial_regular': True,
            'continuidad_regular': True,
        }
    else:
        return {
            'inicial_estacional': False,
            'continuidad_estacional': False,
            'inicial_regular': True,
            'continuidad_regular': True
        }

def build_pipeline_dev_func(eval_periodos, environment, compute, version):
    """
    Builder function to create the pipeline definition.
    This captures build-time configurations (lists, environment strings) that are NOT valid Pipeline Inputs.
    """
    @dsl.pipeline(description="Pipeline Cloud Dev Pipeline")
    def pipeline_dev_func(ult_periodo, n_periodos, all_periodos, min_periodo, mode):
        

        # ---------------------------- ETL -----------------------------------
        # Define ETL Component
        etl_component = command(
            code="./src/data",
            command="python etl.py \
                --input_datastore ${{inputs.input_datastore}} \
                --output_datastore ${{outputs.output_datastore}} \
                --ult_periodo ${{inputs.ult_periodo}} \
                --n_periodos ${{inputs.n_periodos}} \
                --all_periodos ${{inputs.all_periodos}}",
            inputs={
                "input_datastore": Input(type=AssetTypes.URI_FOLDER),
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
        
        # Invoke ETL Step
        etl_step = etl_component(
            input_datastore=Input(type=AssetTypes.URI_FOLDER, path="azureml://datastores/rawdata/paths/rawdata"),
            ult_periodo=ult_periodo,
            n_periodos=n_periodos,
            all_periodos=all_periodos
        )
        etl_step.compute = compute
        etl_step.name = "etl_step"
        etl_step.display_name = "etl_step"

        # ---------------------------- LOOP -----------------------------------
        
        tipos_list = ['continuidad_regular', 'continuidad_estacional', 'inicial_regular', 'inicial_estacional']

        for eval_periodo in eval_periodos:
            
            mapping = get_mapping_tipos(eval_periodo)
            
            for tipo in tipos_list:
                if not mapping[tipo]:
                    continue
                    
                # ---------------------------- FEATURES -----------------------------------
                # Define Features Component
                feat_component_name = f"features_{tipo}_component"
                feat_component = command(
                    code="./src/features",
                    command=f"python {tipo}.py \
                        --input_datastore ${{{{inputs.input_datastore}}}} \
                        --output_feats_train_datastore ${{{{outputs.output_feats_train_datastore}}}} \
                        --output_feats_test_datastore ${{{{outputs.output_feats_test_datastore}}}} \
                        --output_target_test_datastore ${{{{outputs.output_target_test_datastore}}}} \
                        --ult_periodo ${{{{inputs.ult_periodo}}}} \
                        --periodo {eval_periodo}",
                    inputs={
                        "input_datastore": Input(type=AssetTypes.URI_FOLDER),
                        "ult_periodo": Input(type="integer")
                    },
                    outputs={
                        "output_feats_train_datastore": Output(type=AssetTypes.URI_FOLDER),
                        "output_feats_test_datastore": Output(type=AssetTypes.URI_FOLDER),
                        "output_target_test_datastore": Output(type=AssetTypes.URI_FOLDER)
                    },
                    environment=environment,
                    display_name=feat_component_name
                )
                
                # Invoke Features Step
                feat_step_name = f"features_{tipo}_{eval_periodo}"
                feat_step = feat_component(
                    input_datastore=etl_step.outputs.output_datastore,
                    ult_periodo=ult_periodo
                )
                
                # Explicitly map outputs to datastore paths with versioning
                feat_train_path = f"azureml://datastores/platinumdata/paths/features/train/{version}/{tipo}_{eval_periodo}"
                feat_test_path = f"azureml://datastores/platinumdata/paths/features/test/{version}/{tipo}_{eval_periodo}"
                target_test_path = f"azureml://datastores/platinumdata/paths/target/test/{version}/{tipo}_{eval_periodo}"
                
                feat_step.outputs.output_feats_train_datastore = Output(type=AssetTypes.URI_FOLDER, path=feat_train_path)
                feat_step.outputs.output_feats_test_datastore = Output(type=AssetTypes.URI_FOLDER, path=feat_test_path)
                feat_step.outputs.output_target_test_datastore = Output(type=AssetTypes.URI_FOLDER, path=target_test_path)
                
                feat_step.compute = compute
                feat_step.name = feat_step_name
                feat_step.display_name = feat_step_name
                
                # ---------------------------- MODELS -----------------------------------
                # Define Model Component
                model_component_name = f"models_{tipo}_component"
                model_component = command(
                    code="./src/models",
                    command=f"python {tipo}.py \
                        --input_feats_train_datastore ${{{{inputs.input_feats_train_datastore}}}} \
                        --output_model_datastore ${{{{outputs.output_model_datastore}}}} \
                        --periodo {eval_periodo}",
                    inputs={
                        "input_feats_train_datastore": Input(type=AssetTypes.URI_FOLDER)
                    },
                    outputs={
                        "output_model_datastore": Output(type=AssetTypes.URI_FOLDER)
                    },
                    environment=environment,
                    display_name=model_component_name
                )

                # Invoke Model Step
                model_step_name = f"models_{tipo}_{eval_periodo}"
                model_step = model_component(
                    input_feats_train_datastore=feat_step.outputs.output_feats_train_datastore
                )
                
                model_path = f"azureml://datastores/platinumdata/paths/models/{version}/{tipo}_{eval_periodo}"
                model_step.outputs.output_model_datastore = Output(type=AssetTypes.URI_FOLDER, path=model_path)
                
                model_step.compute = compute
                model_step.name = model_step_name
                model_step.display_name = model_step_name
                    
                # ---------------------------- METRICS -----------------------------------
                # Define Metrics Component
                metrics_component_name = f"metrics_{tipo}_component"
                metrics_component = command(
                    code="./src/metrics",
                    command=f"python {tipo}.py \
                        --input_feats_test_datastore ${{{{inputs.input_feats_test_datastore}}}} \
                        --input_model_datastore ${{{{inputs.input_model_datastore}}}} \
                        --input_target_test_datastore ${{{{inputs.input_target_test_datastore}}}} \
                        --output_metrics_datastore ${{{{outputs.output_metrics_datastore}}}} \
                        --output_forecast_datastore ${{{{outputs.output_forecast_datastore}}}} \
                        --periodo_model {eval_periodo} \
                        --periodo {eval_periodo} \
                        --mode ${{{{inputs.mode}}}}",
                    inputs={
                        "input_feats_test_datastore": Input(type=AssetTypes.URI_FOLDER),
                        "input_model_datastore": Input(type=AssetTypes.URI_FOLDER),
                        "input_target_test_datastore": Input(type=AssetTypes.URI_FOLDER),
                        "mode": Input(type="string")
                    },
                    outputs={
                        "output_metrics_datastore": Output(type=AssetTypes.URI_FOLDER),
                        "output_forecast_datastore": Output(type=AssetTypes.URI_FOLDER)
                    },
                    environment=environment,
                    display_name=metrics_component_name
                )

                # Invoke Metrics Step
                metrics_step_name = f"metrics_{tipo}_{eval_periodo}"
                metrics_step = metrics_component(
                    input_feats_test_datastore=feat_step.outputs.output_feats_test_datastore,
                    input_model_datastore=model_step.outputs.output_model_datastore,
                    input_target_test_datastore=feat_step.outputs.output_target_test_datastore,
                    mode=mode
                )
                
                metrics_path = f"azureml://datastores/platinumdata/paths/metrics/{version}/{tipo}_{eval_periodo}"
                forecast_path = f"azureml://datastores/platinumdata/paths/forecast/{version}/{tipo}_{eval_periodo}"
                
                metrics_step.outputs.output_metrics_datastore = Output(type=AssetTypes.URI_FOLDER, path=metrics_path)
                metrics_step.outputs.output_forecast_datastore = Output(type=AssetTypes.URI_FOLDER, path=forecast_path)
                
                metrics_step.compute = compute
                metrics_step.name = metrics_step_name
                metrics_step.display_name = metrics_step_name
    
    return pipeline_dev_func


import re
from azure.storage.blob import BlobServiceClient

def get_next_version(ml_client, datastore_name="platinumdata", path_prefix="paths/features/train"):
    """
    Determines the next version number by scanning the blob container.
    """
    try:
        datastore = ml_client.datastores.get(datastore_name)
        
        # Construct connection string or use account URL + credential
        # Datastore usually has account_name and container_name
        account_name = datastore.account_name
        container_name = datastore.container_name
        
        # We need credentials. DefaultAzureCredential should work if the user has rights.
        credential = DefaultAzureCredential()
        account_url = f"https://{account_name}.blob.core.windows.net"
        
        blob_service_client = BlobServiceClient(account_url, credential=credential)
        container_client = blob_service_client.get_container_client(container_name)
        
        # List blobs with prefix to find folders
        # We are looking for structure: paths/features/train/v{N}/...
        # So we search in 'paths/features/train/' and look for directories that match 'v\d+'
        
        # Blob storage is flat, but we can list with delimiter '/' to simulate folders
        iterator = container_client.walk_blobs(name_starts_with=path_prefix + "/", delimiter="/")
        
        max_version = 0
        
        for item in iterator:
            # item.name is like 'paths/features/train/v1/'
            # valid folders end with /
            if item.name.endswith('/'):
                # Extract the last folder name 
                parts = item.name.strip('/').split('/')
                folder_name = parts[-1] 
                
                match = re.match(r'^v(\d+)$', folder_name)
                if match:
                    version_num = int(match.group(1))
                    if version_num > max_version:
                        max_version = version_num
                        
        next_version = f"v{max_version + 1}"
        return next_version
        
    except Exception as e:
        print(f"Warning: Could not determine next version automatically. Defaulting to v1. Error: {e}")
        return "v1"


if __name__ == '__main__':
    # Configuration
    mode = 'dev'
    eval_periodos = [202505] # Example list
    ult_periodo = max(eval_periodos)
    min_periodo = min(eval_periodos)
    n_periodos = 'None'
    all_periodos = 'True'
    environment = 'docker-image-plus-conda-example:2'
    compute = 'mldevci'
    
    # Initialize ML Client
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    
    # Calculate version
    version = get_next_version(ml_client)
    
    print(f"Creating pipeline job for version {version}...")
    
    # 1. Build the pipeline function with config
    pipeline_dev_func = build_pipeline_dev_func(
        eval_periodos=eval_periodos,
        environment=environment,
        compute=compute,
        version=version
    )
    
    # 2. Instantiate the pipeline with runtime inputs
    pipeline_job = pipeline_dev_func(
        ult_periodo=ult_periodo,
        n_periodos=n_periodos,
        all_periodos=all_periodos,
        min_periodo=min_periodo,
        mode=mode
    )
    
    # Submit via MLClient
    # ml_client already initialized
    
    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="pipeline_cloud_dev_dsl"
    )
    
    print(f"Pipeline submitted: {submitted_job.studio_url}")
