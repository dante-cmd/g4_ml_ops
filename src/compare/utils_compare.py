"""utilities for model comparison"""

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_evaluation_inicial_datastore", dest="input_evaluation_inicial_datastore", type=str)
    parser.add_argument("--input_evaluation_continuidad_horario_datastore", dest="input_evaluation_continuidad_horario_datastore", type=str)
    parser.add_argument("--output_compare_datastore", dest="output_compare_datastore", type=str)
    # parser.add_argument("--feats_version", dest="feats_version", type=str)
    parser.add_argument("--n_eval_periodos", dest="n_eval_periodos", type=int, default=-1)
    parser.add_argument("--model_periodo", dest="model_periodo", type=int)
    parser.add_argument("--model_current_version", dest="model_current_version", type=str)
    parser.add_argument("--model_version", dest="model_version", type=str)
    parser.add_argument("--periodo", dest="periodo", type=int, default=-1)
    # parser.add_argument("--mode", dest="mode", type=str)
    parser.add_argument("--with_tipo", dest="with_tipo", type=str)
    args = parser.parse_args()
    return args


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
            'continuidad_regular_horario':True
            }
        
        return tipos
    elif periodo%100 == 2:
        tipos = {
            'inicial_estacional':True, 
            'continuidad_estacional':True,
            'inicial_regular':True, 
            'continuidad_regular':True, 
            'continuidad_regular_horario':True
            }
        return tipos
    else:
        tipos = {
            'inicial_estacional':False, 
            'continuidad_estacional':False, 
            'inicial_regular':True, 
            'continuidad_regular':True,
            'continuidad_regular_horario':True
            }
        return tipos
    

def get_n_lags(periodo: int, n: int):
    """
    Docstring for get_n_lags
    
    Obtiene el periodo n meses antes del periodo dado.
    :param periodo: Description
    :type periodo: int
    :param n: Description
    :type n: int
    """
    periodo_date = datetime.strptime(str(periodo), '%Y%m')
    return int((periodo_date - relativedelta(months=n)).strftime('%Y%m'))


def get_ahead_n_periodos(periodo: int, n: int):
    """
    Obtiene los n periodos posteriores al periodo dado.
    Ejemplo:
        get_ahead_n_periodos(202306, 3) -> [202306, 202307, 202308]
    """
    return [get_n_lags(periodo, -lag) for lag in range(n)]

