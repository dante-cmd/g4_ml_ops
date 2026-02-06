"""Utils for models"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import argparse


def parse_args():
    """
    Parsea los argumentos de la línea de comandos.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_feats_datastore", dest='input_feats_datastore',
                        type=str)
    parser.add_argument("--output_model_datastore", dest='output_model_datastore',
                        type=str)
    parser.add_argument("--feats_version", dest='feats_version',
                        type=str)
    parser.add_argument("--model_periodo", dest='model_periodo',
                        type=int)
    parser.add_argument("--model_version", dest='model_version',
                        type=str)
    
    parser.add_argument("--mode", dest='mode',
                        type=str)
    parser.add_argument("--with_tipo", dest='with_tipo',
                        type=str)
    
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

