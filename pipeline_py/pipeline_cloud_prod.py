from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from pathlib import Path
import numpy as np
from utils import mk_dirs, run_script
import pandas as pd


def get_mapping_tipos(periodo: int) -> dict:
    """
    Returns a dict that map tipo -> bool type of train to be executed.
    
    Args:
        periodo (int): The period to be trained.
    Returns:
    """
    if periodo%100 == 1:
        tipos = {
            'inicial_estacional':True, 
            'continuidad_estacional':False, 
            'inicial_regular':True, 
            'continuidad_regular':True}
        
        return tipos
    elif periodo%100 == 2:
        tipos = {
            'inicial_estacional':True, 
            'continuidad_estacional':True,
            'inicial_regular':True, 
            'continuidad_regular':True, 
            }
        return tipos
    else:
        tipos = {
            'inicial_estacional':False, 
            'continuidad_estacional':False, 
            'inicial_regular':True, 
            'continuidad_regular':True}
        return tipos


def exec_job_etl(ult_periodo:int, n_periodos:str, all_periodos:str, 
                 environment:str, compute:str):
    
    # --------------------------------------------------------------------
    # ---------------------------- ETL -----------------------------------
    # --------------------------------------------------------------------

    # ult_periodo = 202512
    # n_periodos = "None"
    # all_periodos = "True"
    name = 'etl'
    input_folder_root = input_datastore = 'rawdata'
    output_folder_root = output_datastore = 'platinumdata'
    num_random = np.random.randint(1, 2000)
    display_name = f'{name}_{num_random}'
    experiment_name = f'exp_{name}'

    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    job_inputs = {
        "input_datastore": Input(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{input_folder_root}")}

    job_outputs = {
        "output_datastore": Output(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{output_datastore}/paths/{output_folder_root}")
    }
    
    # configure job
    job = command(
        code="./src/data",
        command="python etl.py \
            --input_datastore ${{inputs.input_datastore}} \
            --output_datastore ${{outputs.output_datastore}}\
            --ult_periodo %d\
            --n_periodos '%s'\
            --all_periodos '%s'"% (ult_periodo, n_periodos, all_periodos),
        inputs=job_inputs,
        outputs=job_outputs,
        environment=environment,
        compute=compute,
        display_name=display_name,
        experiment_name=experiment_name
    )

    # submit job
    returned_job = ml_client.create_or_update(job)
   

def exec_job_features(ult_periodo:int, periodo:int, environment:str, compute:str):

    # --------------------------------------------------------------------
    # ---------------------------- FEATURES ------------------------------
    # --------------------------------------------------------------------


    # ult_periodo = periodo = 202512
    name = 'features'
    input_datastore = 'platinumdata'
    output_feats_train_datastore = 'features/train'
    output_feats_test_datastore = 'features/test'
    output_target_test_datastore = 'target/test'
    num_random = np.random.randint(1, 2000)

    mapping_tipos = get_mapping_tipos(periodo)

    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    job_inputs = {
        "input_datastore": Input(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{input_datastore}")
        }

    job_outputs = {
        "output_feats_train_datastore": Output(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{output_feats_train_datastore}"),
        "output_feats_test_datastore": Output(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{output_feats_test_datastore}"),
        "output_target_test_datastore": Output(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{output_target_test_datastore}")
    }

    tipos  = ['continuidad_regular', 'continuidad_estacional', 
              'inicial_regular', 'inicial_estacional']
    
    for tipo in tipos:
        if mapping_tipos[tipo]:
            display_name = f'{name}_{tipo}_{num_random}'
            experiment_name = f'exp_{name}'
            # 1. Continuidad Regular
            job = command(
                code="./src/features",
                command="python %s.py \
                    --input_datastore ${{inputs.input_datastore}} \
                    --output_feats_train_datastore ${{outputs.output_feats_train_datastore}}\
                    --output_feats_test_datastore ${{outputs.output_feats_test_datastore}}\
                    --output_target_test_datastore ${{outputs.output_target_test_datastore}}\
                    --ult_periodo %d\
                    --periodo '%s'"% (tipo, ult_periodo, periodo),
                inputs=job_inputs,
                outputs=job_outputs,
                environment=environment,
                compute=compute,
                display_name=display_name,
                experiment_name=experiment_name
            )

            # submit job
            returned_job = ml_client.create_or_update(job)


def exec_job_models(periodo:int, environment:str, compute:str):

    name = 'models'
    input_datastore = 'platinumdata'
    input_feats_train_datastore = 'features/train'
    output_model_datastore = 'models'

    num_random = np.random.randint(1, 2000)

    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    mapping_tipos = get_mapping_tipos(periodo)

    job_inputs = {
        "input_feats_train_datastore": Input(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{input_feats_train_datastore}")
        }

    job_outputs = {
        "output_model_datastore": Output(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{output_model_datastore}")
    }

    tipos  = ['continuidad_regular', 'continuidad_estacional', 
              'inicial_regular', 'inicial_estacional']
    
    for tipo in tipos:
        if mapping_tipos[tipo]:
            display_name = f'{name}_{tipo}_{num_random}'
            experiment_name = f'exp_{name}'
            # 1. Continuidad Regular
            job = command(
                code="./src/models",
                command="python %s.py \
                    --input_feats_train_datastore ${{inputs.input_feats_train_datastore}} \
                    --output_model_datastore ${{outputs.output_model_datastore}}\
                    --periodo '%s'"% (tipo, periodo),
                inputs=job_inputs,
                outputs=job_outputs,
                environment=environment,
                compute=compute,
                display_name=display_name,
                experiment_name=experiment_name
            )

            # submit job
            returned_job = ml_client.create_or_update(job)


def exec_job_metrics(periodo:int, compute:str, environment:str, mode:str):

    # periodo = 202512
    name = 'metrics'
    input_datastore = 'platinumdata'
    input_feats_test_datastore = 'features/test'
    input_model_datastore = 'models'
    input_target_test_datastore = 'target/test'
    output_metrics_datastore = 'metrics'
    
    mapping_tipos = get_mapping_tipos(periodo)

    num_random = np.random.randint(1, 2000)

    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    job_inputs = {
        "input_feats_test_datastore": Input(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{input_feats_test_datastore}"),
        "input_model_datastore": Input(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{input_model_datastore}"),
        "input_target_test_datastore": Input(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{input_target_test_datastore}")
        }

    job_outputs = {
        "output_metrics_datastore": Output(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{output_metrics_datastore}")
    }

    tipos  = ['continuidad_regular', 'continuidad_estacional', 
              'inicial_regular', 'inicial_estacional']
    
    for tipo in tipos:
        if mapping_tipos[tipo]:
            display_name = f'{name}_{tipo}_{num_random}'
            experiment_name = f'exp_{name}'
             # 1. Continuidad Regular
            job = command(
                code="./src/metrics",
                command="python %s.py \
                    --input_feats_test_datastore ${{inputs.input_feats_test_datastore}} \
                    --input_model_datastore ${{inputs.input_model_datastore}} \
                    --input_target_test_datastore ${{inputs.input_target_test_datastore}} \
                    --output_metrics_datastore ${{outputs.output_metrics_datastore}} \
                    --periodo '%s' \
                    --mode '%s'"% (tipo, periodo, mode),
                inputs=job_inputs,
                outputs=job_outputs,
                environment=environment,
                compute=compute,
                display_name=display_name,
                experiment_name=experiment_name
            )
            returned_job = ml_client.create_or_update(job)
            
    
if __name__ == '__main__':
    # For dev ult_periodo = periodo
    # For prod ult_periodo = Lag{1}(periodo)
    # mode = dev, prod }
    mode = 'dev'
    periodo = 202501
    ult_periodo = 202501
    n_periodos = 'None'
    all_periodos = 'True'
    environment = ''
    compute = ''
    # mk_dirs()
    exec_job_etl(
        ult_periodo=ult_periodo, 
        n_periodos=n_periodos, 
        all_periodos=all_periodos, 
        environment=environment, 
        compute=compute)
    
    exec_job_features(ult_periodo=ult_periodo, 
                      periodo=periodo,
                      environment=environment,
                      compute=compute)
    exec_job_models(
        periodo=periodo, 
        environment=environment, 
        compute=compute)
    
    exec_job_metrics(
        periodo=periodo, 
        compute=compute, 
        environment=environment, 
        mode=mode)
