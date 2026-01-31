from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import argparse
# import mlflow
# from mlflow.tracking import MlflowClient


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_model_datastore", dest="input_model_datastore", type=str)
    parser.add_argument("--input_feats_datastore", dest="input_feats_datastore", type=str)
    # parser.add_argument("--input_target_datastore", dest="input_target_datastore", type=str)
    parser.add_argument("--output_predict_datastore", dest="output_predict_datastore", type=str)
    parser.add_argument("--feats_version", dest="feats_version", type=str)
    parser.add_argument("--target_version", dest="target_version", type=str)
    parser.add_argument("--periodo", dest="periodo", type=int)
    parser.add_argument("--model_periodo", dest="model_periodo", type=int)
    parser.add_argument("--model_version", dest="model_version", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# Optional: Set tracking URI if not using default
# mlflow.set_tracking_uri("http://your-mlflow-server:5000")

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
            'continuidad_regular':True,
            'continuidad_regular_horario':True}
        
        return tipos
    elif periodo%100 == 2:
        tipos = {
            'inicial_estacional':True, 
            'continuidad_estacional':True,
            'inicial_regular':True, 
            'continuidad_regular':True, 
            'continuidad_regular_horario':True}
        return tipos
    else:
        tipos = {
            'inicial_estacional':False, 
            'continuidad_estacional':False, 
            'inicial_regular':True, 
            'continuidad_regular':True,
            'continuidad_regular_horario':True}
        return tipos

# model_name = "YourModelName"
# tag_key = "stage"
# tag_value = "Production"  # or any value like "staging", "latest", etc.
def get_n_lags(periodo: int, n: int):
    periodo_date = datetime.strptime(str(periodo), '%Y%m')
    return int((periodo_date - relativedelta(months=n)).strftime('%Y%m'))


def get_ahead_n_periodos(periodo: int, n: int):
    """
    Obtiene los n periodos posteriores al periodo dado.
    Ejemplo:
        get_ahead_n_periodos(202306, 3) -> [202306, 202307, 202308]
    """
    return [get_n_lags(periodo, -lag) for lag in range(n)]

