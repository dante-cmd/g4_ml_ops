
"""
Utils Features - Utilidades para el feature engineering
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import argparse


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


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()
    
    # add arguments
    parser.add_argument("--input_datastore", dest='input_datastore',
                        type=str)
    parser.add_argument("--output_feats_datastore", dest='output_feats_datastore',
                        type=str)
    parser.add_argument("--output_target_datastore", dest='output_target_datastore',
                        type=str)
    parser.add_argument("--platinum_version", dest='platinum_version',
                        type=str)
    parser.add_argument("--feats_version", dest='feats_version',
                        type=str)
    parser.add_argument("--target_version", dest='target_version',
                        type=str)
    parser.add_argument("--n_eval_periodos", dest='n_eval_periodos',
                        type=int, default=-1)
    parser.add_argument("--ult_periodo", dest='ult_periodo',
                        type=int)
    parser.add_argument("--model_periodo", dest='model_periodo',
                        type=int)
    parser.add_argument("--mode", dest='mode',
                        type=str)
    parser.add_argument("--periodo", dest='periodo',
                        type=int, default=-1)
    parser.add_argument("--with_tipo", dest='with_tipo',
                        type=str)
    

    # parse args
    args = parser.parse_args()

    # return args
    return args


def get_n_lags(periodo: int, n: int):
    periodo_date = datetime.strptime(str(periodo), '%Y%m')
    return int((periodo_date - relativedelta(months=n)).strftime('%Y%m'))


def get_last_n_periodos(periodo: int, n: int):
    """
    Obtiene los n periodos anteriores al periodo dado.
    Ejemplo:
        get_last_n_periodos(202306, 3) -> [202306, 202305, 202304]
    """
    return [get_n_lags(periodo, lag) for lag in range(n)]


def get_ahead_n_periodos(periodo: int, n: int):
    """
    Obtiene los n periodos posteriores al periodo dado.
    Ejemplo:
        get_ahead_n_periodos(202306, 3) -> [202306, 202307, 202308]
    """
    return [get_n_lags(periodo, -lag) for lag in range(n)]


def get_all_periodos(periodo: int):
    periodos = pd.period_range(
            start=datetime(2022, 11, 1),
            end=datetime(periodo // 100, periodo % 100, 1),
            freq='M')
    periodos = periodos.strftime('%Y%m').astype('int32')
    return periodos


def validate_periodos(n_periodos:int|None, ult_periodo:int, all_periodos:bool):
    if n_periodos is not None:
        periodos = get_last_n_periodos(ult_periodo, n_periodos)
        min_periodo = int(np.min(periodos))
        print("Rango de actulización:",  min_periodo, "-", ult_periodo)
        # print(f"Prog Acad se está actualizando desde {min_periodo} hasta {ult_periodo}")
    else:
        assert all_periodos
        periodos = get_all_periodos(ult_periodo).copy()
        min_periodo = int(np.min(periodos))
        print("Rango de actulización:",  min_periodo, "-", ult_periodo)
        # print(f"Prog Acad se está actualizando desde {min_periodo} hasta {ult_periodo}")
    return periodos


def get_training_periodos(periodo: int):
    """
    Returns a list of training periods for a given period.
    The minimum period is 202401 and the maximum period is the given period.    
    Args:
        periodo (int): The period for which to generate training periods.
    
    Returns:
        list: A list of training periods.
    """
    periodos = pd.period_range(
            start=datetime(2024, 3, 1),
            end=datetime(periodo // 100, periodo % 100, 1),
            freq='M')
    periodos = periodos.strftime('%Y%m').astype('int32')
    return list(periodos)


def get_training_periodos_estacionales(periodo: int, meses: list[int]):
    periodos = get_training_periodos(periodo)
    periodos_selected = [per for per in periodos if per % 100 in meses]
    return periodos_selected


def filter_by_hora_atencion(df_idx: pd.DataFrame, df_turno_disponible: pd.DataFrame, df_horario: pd.DataFrame):
    df_idx_clone = df_idx.copy()

    df_idx_clone_01 = df_idx_clone.merge(
        df_horario[['HORARIO', 'HORA_MIN', 'HORA_MAX']].rename(
            columns={'HORARIO': 'HORARIO_ACTUAL'}
        ),
        on=['HORARIO_ACTUAL'],
        how='left'
    )
    
    assert df_idx_clone_01['HORA_MIN'].isnull().sum() == 0
    assert df_idx_clone_01['HORA_MAX'].isnull().sum() == 0

    # df_idx_clone_02 = df_idx_clone_01.copy()
    df_idx_clone_02 = df_idx_clone_01.rename(
        columns={'HORA_MIN': 'HORA_INICIO', 'HORA_MAX': 'HORA_FIN'}
    )

    # df_idx_clone['HORA_INICIO'] = (
    #     df_idx_clone['HORARIO_ACTUAL'].str.replace(PATTERN_HORARIO, r"\1", regex=True).astype('int32') +
    #     df_idx_clone['HORARIO_ACTUAL'].str.replace(PATTERN_HORARIO, r"\2", regex=True).astype('int32')/60)
    # 
    # df_idx_clone['HORA_FIN'] = (
    #     df_idx_clone['HORARIO_ACTUAL'].str.replace(PATTERN_HORARIO, r"\3", regex=True).astype('int32') +
    #     df_idx_clone['HORARIO_ACTUAL'].str.replace(PATTERN_HORARIO, r"\4", regex=True).astype('int32')/60)
            
    df_turno_disponible_01 = df_turno_disponible.rename(
        columns={'PERIODO': 'PERIODO_TARGET'}
    ).copy()
            
    df_idx_clone_03 = df_idx_clone_02.merge(
        df_turno_disponible_01,
        on=['PERIODO_TARGET', 'SEDE'],
        how='left'
    )

    # if df_idx_clone_01['HORA_MAX'].isnull().sum() == 0:
    #     pass
    # else:
    #     ww = df_idx_clone_01[df_idx_clone_01['HORA_MAX'].isnull()].copy()
    #     ww.to_excel('output/ww.xlsx', index=False)
    assert df_idx_clone_03['HORA_MAX'].isnull().sum() == 0

    filtro = (
        (df_idx_clone_03['HORA_MAX'] >= df_idx_clone_03['HORA_FIN']) & 
        (df_idx_clone_03['HORA_MIN'] <= df_idx_clone_03['HORA_INICIO'])
    )

    df_idx_clone_04 = df_idx_clone_03[filtro].copy()
    
    return df_idx_clone_04

