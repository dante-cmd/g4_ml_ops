"""Utils for models"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_classes(df_forecast:pd.DataFrame):
    """
    Apply rules to forecast in order to compute the number of classes predicted from the number of students predicted. 
    
    Args:
        df_forecast (pd.DataFrame): DataFrame with forecast data.
        
    Returns:
        pd.DataFrame: DataFrame with forecast data after applying rules.
        
    Notes:
        - The function applies rules to calculate the number of classes predicted
          from the number of students predicted.
            - The formula is:
              CANT_CLASES_PREDICT = (CANT_ALUMNOS_PREDICT // VAC_ACAD_ESTANDAR +
                                     np.where((CANT_ALUMNOS_PREDICT % VAC_ACAD_ESTANDAR) >= PE, 1, 0))
              if CANT_ALUMNOS_PREDICT > 0, otherwise 0
    """

    df_forecast_01 = df_forecast.copy()
    df_forecast_01['CANT_CLASES_PREDICT'] = (
            np.where(
                    df_forecast_01['CANT_ALUMNOS_PREDICT'] > 0,
                    (df_forecast_01['CANT_ALUMNOS_PREDICT'] // df_forecast_01['VAC_ACAD_ESTANDAR'] +
                     np.where(
                     (df_forecast_01['CANT_ALUMNOS_PREDICT'] %
                      df_forecast_01['VAC_ACAD_ESTANDAR']) >= df_forecast_01['PE'],
                         1, 0)), 0)
    )

    return df_forecast_01


def join_target(df_forecast: pd.DataFrame, df_target:pd.DataFrame, on_cols: list[str]):
    """
    Join target data to forecast data.
        
    Args:
        df_forecast (pd.DataFrame): DataFrame with forecast data.
        
    Returns:
        pd.DataFrame: DataFrame with forecast data after joining target data.
    """

    df_forecast_01 = df_forecast.merge(
        df_target,
        on=on_cols,
        how='outer'
    )
    
    df_forecast_01['CANT_CLASES'] = df_forecast_01['CANT_CLASES'].fillna(0).astype('int32')
    df_forecast_01['CANT_ALUMNOS'] = df_forecast_01['CANT_ALUMNOS'].fillna(0).astype('int32')
    df_forecast_01['CANT_ALUMNOS_PREDICT'] = df_forecast_01['CANT_ALUMNOS_PREDICT'].fillna(0).astype('float64')
    df_forecast_01['CANT_CLASES_PREDICT'] = df_forecast_01['CANT_CLASES_PREDICT'].fillna(0).astype('float64')
    df_forecast_01['IDX'] = df_forecast_01['IDX'].fillna(0).astype('int32')

    return df_forecast_01


def calculate_metrics(df_forecast:pd.DataFrame, df_target:pd.DataFrame, on_cols: list[str]):
    """
    Calcula las métricas del modelo.
    
    Args:
        df_forecast (pd.DataFrame): DataFrame con forecast data.
        df_target (pd.DataFrame): DataFrame con target data.
        on_cols (list[str]): Columnas para join.
        
    Returns:
        dict: Diccionario con las métricas calculadas.
    """
    
    # logging.info(f' Mode: {mode} {periodo}'.center(50, '=').upper())
    # logging.info(f'Tipo: {tipo}'.upper())

    df_forecast_02 = join_target(df_forecast, df_target, on_cols)

    df_forecast_formado = df_forecast_02[
            df_forecast_02['CANT_CLASES'] > 0].copy()
    pct_comb = df_forecast_formado['IDX'].mean()
    total_comb = df_forecast_formado['IDX'].sum()
    
    # logging.info("% Comb: {:.1%}".format(pct_comb))
    # logging.info("# Comb: {:n}".format(total_comb))
    
    # df_forecast_02 = self.calculate_classes(df_forecast_01)
    es_zero_predict = df_forecast_02['CANT_CLASES_PREDICT'] == 0
    es_zero = df_forecast_02['CANT_CLASES'] == 0
    df_forecast_03 = df_forecast_02[~(es_zero_predict & es_zero)].copy()
    faltante = np.where(
            df_forecast_03['CANT_CLASES'] > df_forecast_03['CANT_CLASES_PREDICT'],
            df_forecast_03['CANT_CLASES'] - df_forecast_03['CANT_CLASES_PREDICT'],
            0)
    sobrante = np.where(
            df_forecast_03['CANT_CLASES_PREDICT'] > df_forecast_03['CANT_CLASES'],
            df_forecast_03['CANT_CLASES_PREDICT'] - df_forecast_03['CANT_CLASES'],
            0)
    # logging.info("Total Faltante: {:n}".format(faltante.sum()))
    # logging.info("Total Sobrante: {:n}".format(sobrante.sum()))
    # logging.info("Total Gestion: {:n}".format(faltante.sum() + sobrante.sum()))
    match_clases = np.where(
            df_forecast_03['CANT_CLASES'] == df_forecast_03['CANT_CLASES_PREDICT'], 1, 0)
    precision = match_clases.mean()
    
    # logging.info("% Precision {:.1%}".format(precision))
    # logging.info("Total Clases (vs Predict): {:,.0f} ({:,.0f})".format(
    #         df_forecast_03['CANT_CLASES'].sum(), df_forecast_03['CANT_CLASES_PREDICT'].sum()))
    # logging.info("Total alumnos (vs Predict): {:,.0f} ({:,.0f})".format(
    #         df_forecast_03['CANT_ALUMNOS'].sum(), df_forecast_03['CANT_ALUMNOS_PREDICT'].sum()))
    return {
        "pct_comb": pct_comb,
        "total_comb": total_comb,
        "faltante": faltante.sum(),
        "sobrante": sobrante.sum(),
        "precision": precision,
        "total_clases": df_forecast_03['CANT_CLASES'].sum(),
        "total_alumnos": df_forecast_03['CANT_ALUMNOS'].sum(),
        "total_alumnos_predict": df_forecast_03['CANT_ALUMNOS_PREDICT'].sum()
    }


def fac_to_cant(df_forecast_fac: pd.DataFrame, df_forecast_cant:pd.DataFrame) -> pd.DataFrame:
    """
    Get the number of students from the forecast data.

    Args:
        df_forecast (pd.DataFrame): DataFrame with forecast data.
        df_predict (pd.DataFrame): DataFrame with forecast data.
    Returns:
        pd.DataFrame: DataFrame with forecast data after getting the number of students.
    """
        
    df_forecast_fac_01 = df_forecast_fac[
            ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL', 'IDX',
             'PE', 'VAC_ACAD_ESTANDAR', 'FAC_ALUMNOS_PREDICT', 'CANT_ALUMNOS_ANTERIOR', 'CANT_CLASES_ANTERIOR']].copy()
        
    df_forecast_cant_01 = df_forecast_cant.loc[
            df_forecast_cant.CANT_ALUMNOS_PREDICT >= df_forecast_cant.PE,
            ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'CANT_ALUMNOS_PREDICT']].copy()
        
    df_forecast_cant_02 = df_forecast_cant_01.merge(
            df_forecast_fac_01,
            on=['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL'],
            how='left'
        )

    assert df_forecast_cant_02['FAC_ALUMNOS_PREDICT'].isnull().sum() == 0
        
    collection = []
    
    for _, data in df_forecast_cant_02.groupby(['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL']):
        fac_alumnos_predict =  np.asarray(data['FAC_ALUMNOS_PREDICT'])

        if np.all(fac_alumnos_predict.sum() == 0):
            fac_alumnos_predict[:] = 1
        
        fac_alumnos_predict = fac_alumnos_predict / fac_alumnos_predict.sum()
        total_alumnos_predict =  np.asarray(data['CANT_ALUMNOS_PREDICT'])
        # data_01 = data.reset_index(drop=True)
        cant_alumnos_predict = (total_alumnos_predict * fac_alumnos_predict)
        quotient = cant_alumnos_predict//1
        remainder = cant_alumnos_predict%1
        total_remainder = int(round(remainder.sum()))
        zeros = np.zeros_like(quotient)
        if total_remainder > 0:
            order = np.flip(np.argsort(remainder))
            zeros[order[:total_remainder]] = 1
        cant_alumnos = quotient + zeros
        data['CANT_ALUMNOS'] = cant_alumnos
        # collection.append(data)
        
        if data.shape[0] == 1:
            collection.append(data)
        else:
            data_lt_pe = data[data['CANT_ALUMNOS'] < data['PE']].copy()
            data_gt_pe= data[data['CANT_ALUMNOS'] >= data['PE']].copy()

            total_lt_pe = np.sum(data_lt_pe['CANT_ALUMNOS'])
            
            if total_lt_pe == 0:
                collection.append(data_gt_pe)
            else:
                factor = np.asarray(data_gt_pe['CANT_ALUMNOS']/data_gt_pe['CANT_ALUMNOS'].sum())

                cant_alumnos_predict_01 = np.asarray(factor * total_lt_pe)
                quotient_01 = cant_alumnos_predict_01//1
                remainder_01 = cant_alumnos_predict_01%1
                total_remainder_01 = int(round(remainder_01.sum()))
                zeros_01 = np.zeros_like(quotient_01)
                
                if total_remainder_01 > 0:
                    order_01 = np.flip(np.argsort(remainder_01))
                    zeros_01[order_01[:total_remainder_01]] = 1
                
                data_gt_pe['CANT_ALUMNOS'] = data_gt_pe['CANT_ALUMNOS'] +   quotient_01 + zeros_01
                collection.append(data_gt_pe)
                # data_01['CANT_ALUMNOS'] = cant_alumnos

        
    df_forecast_03 = pd.concat(collection, ignore_index=True)
    df_forecast_03 = df_forecast_03.drop(columns=['CANT_ALUMNOS_PREDICT'])
    df_forecast_03 = df_forecast_03.rename(columns={'CANT_ALUMNOS': 'CANT_ALUMNOS_PREDICT'})

    return df_forecast_03
