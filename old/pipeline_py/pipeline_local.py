import pandas as pd
# from src.data.etl import Etl
# from src.features.loader import Loader as DataTrain
# calculate_classes, calculate_metrics, join_target,  save_output,
from pathlib import Path
from old.utils import mk_dirs, run_script


def get_mapping_tipos(periodo_string: str) -> dict:
    """
    Returns a dict that map tipo -> bool type of train to be executed.
    
    Args:
        periodo (int): The period to be trained.
    Returns:
    """
    
    periodo = int(periodo_string)

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


def get_tipo_train(periodo: int) -> list[bool]:
    """
    Returns a list of booleans indicating the type of train to be executed.
    
    Args:
        periodo (int): The period to be trained.
    
    Returns:
        list[bool]: A list of booleans indicating the type of train to be executed.
        [estacional_inicial, estacional_continuidad, regular_inicial, regular_continuidad]
    """
    if periodo%100 == 1:
        return [True, False, True, True]
    elif periodo%100 == 2:
        return [True, True, True, True]
    else:
        return [False, False, True, True]


def exec_job_etl(ult_periodo:str, n_periodos:str, all_periodos:str, raw_version:str, platinum_version:str):
    
    # --------------------------------------------------------------------
    # ---------------------------- ETL -----------------------------------
    # --------------------------------------------------------------------

    script_to_run = "./src/data/etl.py"
    args = [
    "--input_datastore", f'./data/base_data/rawdata/{raw_version}/', 
    "--output_datastore", f'./data/ml_data/platinumdata/{platinum_version}/', 
    "--ult_periodo", ult_periodo,
    "--n_periodos", n_periodos, 
    "--all_periodos", all_periodos]    
    
    run_script(script_to_run, args)
   

def exec_job_features(periodo:str, ult_periodo:str, 
                        platinum_version:str, 
                        features_version:str, 
                        target_version:str):
    
    args = [
    "--input_datastore", f'./data/ml_data/platinumdata/{platinum_version}/', 
    "--output_feats_train_datastore", f"./data/ml_data/features/{features_version}/train/",
    "--output_feats_test_datastore", f"./data/ml_data/features/{features_version}/test/",
    "--output_target_test_datastore", f"./data/ml_data/target/{target_version}/test/",
    "--periodo", periodo,
    "--ult_periodo", ult_periodo]

    mapping_tipos = get_mapping_tipos(periodo)

    tipos = ["continuidad_regular", "inicial_regular",
             "continuidad_estacional", "inicial_estacional"]
    
    for tipo in tipos:
        if mapping_tipos[tipo]:
            script = f"./src/features/{tipo}.py"
            run_script(script, args)


def exec_job_models(periodo:str, features_version:str, target_version:str, experiment_name:str):

    args = [
    "--input_feats_train_datastore", f"./data/ml_data/features/{features_version}/train/",
    "--input_feats_test_datastore", f"./data/ml_data/features/{features_version}/test/",
    "--input_target_test_datastore", f"./data/ml_data/target/{target_version}/test/",
    "--experiment_name", experiment_name,
    "--periodo", periodo]

    mapping_tipos = get_mapping_tipos(periodo)

    tipos = ["continuidad_regular", "inicial_regular",
             "continuidad_estacional", "inicial_estacional"]
    for tipo in tipos:
        if mapping_tipos[tipo]:
            script = f"./src/models/{tipo}.py"
            run_script(script, args)

def static_evaluation(
    periodos:list[int], 
    ult_periodo:str,
    n_periodos:str, 
    all_periodos:str,
    raw_version:str, 
    platinum_version:str, 
    features_version:str, 
    target_version:str, 
    experiment_name:str):

    exec_job_etl(
            ult_periodo=str(ult_periodo), 
            n_periodos=n_periodos, 
            all_periodos=all_periodos, 
            raw_version=raw_version, 
            platinum_version=platinum_version,
            # environment=environment, 
            # compute=compute
            )
      
    for periodo in periodos:
        exec_job_features(ult_periodo=str(periodo), 
                        periodo=str(periodo),
                        platinum_version=platinum_version,
                        features_version=features_version,
                        target_version=target_version,
                        # environment=environment,
                        # compute=compute
                        )
        exec_job_models(
            periodo=str(periodo), 
            features_version=features_version,
            target_version=target_version,
            experiment_name=experiment_name,
            # environment=environment, 
            # compute=compute
            )


def recent_evaluation(
        periodos:list[int], ult_periodo:int,
        n_periodos:str, all_periodos:str,
        raw_version:str, platinum_version:str, 
        features_version:str, target_version:str, 
        experiment_name:str):

    exec_job_etl(
            ult_periodo=str(ult_periodo), 
            n_periodos=n_periodos, 
            all_periodos=all_periodos, 
            raw_version=raw_version, 
            platinum_version=platinum_version,
            # environment=environment, 
            # compute=compute
            )

    min_periodo = min(periodos)

    for periodo in periodos:
        exec_job_features(
                        ult_periodo=str(periodo), 
                        periodo=str(periodo),
                        platinum_version=platinum_version,
                        features_version=features_version,
                        target_version=target_version,
                        # environment=environment,
                        # compute=compute
                        )
        exec_job_models(
            periodo=str(periodo), 
            features_version=features_version,
            target_version=target_version,
            experiment_name=experiment_name,
            # environment=environment, 
            # compute=compute
            )



if __name__ == '__main__':
    # mk_dirs()
    periodos:list[int] = [202506, 202507, 202508, 202509, 202510, 202511]
    ult_periodo = max(periodos)
    # min_periodo = min(periodos)
    n_periodos = 'None'
    all_periodos = 'True'
    raw_version = 'v1'
    platinum_version = 'v1'
    features_version = 'v1'
    target_version = 'v1'
    experiment_name = 'recent'
    
    recent_evaluation(
        periodos=periodos,
        ult_periodo=ult_periodo,
        n_periodos=n_periodos,
        all_periodos=all_periodos,
        raw_version=raw_version,
        platinum_version=platinum_version,
        features_version=features_version,
        target_version=target_version,
        experiment_name=experiment_name
    )