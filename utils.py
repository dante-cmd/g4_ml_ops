# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
# from azure.ai.ml import MLClient
# from azure.ai.ml import command
from pathlib import Path
# import pandas as pd
# import numpy as np
import subprocess
import sys


# PATTERN_HORARIO = r"(\d{2})\:(\d{2}) - (\d{2})\:(\d{2})"

def mk_dirs():
    base_path = Path('./data')
    if not base_path.exists():
        base_path.mkdir()
    
    if not (base_path/'features').exists():
        (base_path/'features').mkdir()
    
    if not (base_path/'target').exists():
        (base_path/'target').mkdir()

    if not (base_path/'features'/'train').exists():
        (base_path/'features'/'train').mkdir()

    if not (base_path/'features'/'test').exists():
        (base_path/'features'/'test').mkdir()
        pass
    
    if not (base_path/'target'/'test').exists():
        (base_path/'target'/'test').mkdir()
        pass
    
    if not (base_path/'models').exists():
        (base_path/'models').mkdir()

    if not (base_path/'metrics').exists():
        (base_path/'metrics').mkdir()

    if not (base_path/'forecast').exists():
        (base_path/'forecast').mkdir()


def run_script(script_to_run, args:list):
    """
    Executes a single PowerShell command and returns the result.
    
   script_to_run = "./src/data/etl.py"
    args = [
    "--input_datastore", './data/rawdata', 
    "--output_datastore", './data/platinumdata', 
    "--ult_periodo", "202501",
    "--n_periodos", 'None', 
    "--all_periodos", 'True']
    """
    process = subprocess.Popen(
    [sys.executable, script_to_run] + args,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

    print(f"Process started with PID: {process.pid}")
    # You can do other work here while the script runs...

    # Wait for the process to finish and capture output
    stdout, stderr = process.communicate()
    return_code = process.returncode
    # Use 'powershell' (or 'pwsh' for PowerShell 7+) as the executable
    # and pass the command with the '-Command' argument.
    if return_code == 0:
        print(f"Script finished. Output:\n{stdout}")
    else:
        print(f"Script failed with code {return_code}. Error:\n{stderr}")


# def get_n_lags(periodo: int, n: int):
#     periodo_date = datetime.strptime(str(periodo), '%Y%m')
#     return int((periodo_date - relativedelta(months=n)).strftime('%Y%m'))


# def get_last_n_periodos(periodo: int, n: int):
#     return [get_n_lags(periodo, lag) for lag in range(n)]


# def get_all_periodos(periodo: int):
#     periodos = pd.period_range(
#             start=datetime(2022, 11, 1),
#             end=datetime(periodo // 100, periodo % 100, 1),
#             freq='M')
#     periodos = periodos.strftime('%Y%m').astype('int32')
#     return periodos


# # def save_output(mode:str, periodo:int, data_frame:pd.DataFrame, 
# #                      name_file:str):
    
# #     year = str(periodo//100)
# #     periodo_string = str(periodo)
    
# #     path_output = Path(f'output/{mode}')
    
# #     if not path_output.exists():
# #         path_output.mkdir()
    
# #     if not (path_output/year).exists():
# #         (path_output/year).mkdir()
    
# #     if not (path_output/year/periodo_string).exists():
# #         (path_output/year/periodo_string).mkdir()
    
# #     data_frame.to_excel((path_output/year/periodo_string/(periodo_string + f'_{mode}_' + name_file + '.xlsx')), index=False)


# def validate_periodos(n_periodos:int|None, ult_periodo:int, all_periodos:bool):
#     if n_periodos is not None:
#         periodos = get_last_n_periodos(ult_periodo, n_periodos)
#         print("✅ Prog Acad se está actualizando para:", periodos)
#     else:
#         assert all_periodos
#         periodos = get_all_periodos(ult_periodo).copy()
#         print("✅ Prog Acad se está actualizando para:", periodos)
#     return periodos


# def get_training_periodos(periodo: int):
#     """
#     Returns a list of training periods for a given period.
#     The minimum period is 202401 and the maximum period is the given period.    
#     Args:
#         periodo (int): The period for which to generate training periods.
    
#     Returns:
#         list: A list of training periods.
#     """
#     periodos = pd.period_range(
#             start=datetime(2024, 1, 1),
#             end=datetime(periodo // 100, periodo % 100, 1),
#             freq='M')
#     periodos = periodos.strftime('%Y%m').astype('int32')
#     return list(periodos)


# def get_training_periodos_estacionales(periodo: int, meses: list[int]):
#     periodos = get_training_periodos(periodo)
#     periodos_selected = [per for per in periodos if per % 100 in meses]
#     return periodos_selected


# # def calculate_classes(df_forecast:pd.DataFrame):
# #     """
# #     Apply rules to forecast in order to compute the number of classes predicted from the number of students predicted. 
    
# #     Args:
# #         df_forecast (pd.DataFrame): DataFrame with forecast data.
        
# #     Returns:
# #         pd.DataFrame: DataFrame with forecast data after applying rules.
        
# #     Notes:
# #         - The function applies rules to calculate the number of classes predicted
# #           from the number of students predicted.
# #             - The formula is:
# #               CANT_CLASES_PREDICT = (CANT_ALUMNOS_PREDICT // VAC_ACAD_ESTANDAR +
# #                                      np.where((CANT_ALUMNOS_PREDICT % VAC_ACAD_ESTANDAR) >= PE, 1, 0))
# #               if CANT_ALUMNOS_PREDICT > 0, otherwise 0
# #     """

# #     # df_forecast['CANT_CLASES_PREDICT']
# #     df_forecast_01 = df_forecast.copy()
# #     df_forecast_01['CANT_CLASES_PREDICT'] = (
# #             np.where(
# #                     df_forecast_01['CANT_ALUMNOS_PREDICT'] > 0,
# #                     (df_forecast_01['CANT_ALUMNOS_PREDICT'] // df_forecast_01['VAC_ACAD_ESTANDAR'] +
# #                      np.where(
# #                      (df_forecast_01['CANT_ALUMNOS_PREDICT'] %
# #                       df_forecast_01['VAC_ACAD_ESTANDAR']) >= df_forecast_01['PE'],
# #                          1, 0)), 0)
# #     )

# #     return df_forecast_01


# # def calculate_alumnos(df_forecast:pd.DataFrame):
# #     """
# #     Apply rules to forecast in order to compute the number of students predicted from the number of classes predicted. 
    
# #     Args:
# #         df_forecast (pd.DataFrame): DataFrame with forecast data.
        
# #     Returns:
# #         pd.DataFrame: DataFrame with forecast data after applying rules.
        
# #     Notes:
# #         - The function applies rules to calculate the number of students predicted
# #           from the number of classes predicted.
# #             - The formula is:
# #               CANT_ALUMNOS_PREDICT = (CANT_CLASES_PREDICT * VAC_ACAD_ESTANDAR)
# #               if CANT_CLASES_PREDICT > 0, otherwise 0
# #     """
    
# #     # df_forecast['CANT_ALUMNOS_PREDICT']
# #     df_forecast_01 = df_forecast.copy()
# #     cant_clases_predict = np.asarray(df_forecast_01['CANT_CLASES_PREDICT'])
# #     cant_alumnos_predict = np.asarray(df_forecast_01['CANT_ALUMNOS_PREDICT'])
# #     vac_acad_estandar = np.asarray(df_forecast_01['VAC_ACAD_ESTANDAR'])

# #     new_cant_alumnos_predict = (
# #             np.where(
# #                     cant_clases_predict > 0,
# #                     np.minimum(
# #                         cant_clases_predict * vac_acad_estandar,
# #                         cant_alumnos_predict), 
# #                     0))

# #     faltante = cant_alumnos_predict - new_cant_alumnos_predict
# #     espacio = np.where(
# #         (cant_clases_predict>0) & (faltante == 0),
# #         cant_clases_predict * vac_acad_estandar - new_cant_alumnos_predict,
# #         0)
        
# #     assert espacio.sum() >= faltante.sum(), "Error en la asignación de espacio"
    
# #     fac_espacio = espacio/espacio.sum()
# #     quotient = (faltante*fac_espacio) // 1
# #     remainder = (faltante*fac_espacio) % 1
# #     total_remainder = int(round(remainder.sum()))

# #     zeros = np.zeros_like(quotient)
# #     if total_remainder > 0:
# #         order = np.flip(np.argsort(remainder))
# #         zeros[order[:total_remainder]] = 1

# #     df_forecast_01['CANT_ALUMNOS_PREDICT'] = new_cant_alumnos_predict + quotient + zeros
    
# #     return df_forecast_01


# # def join_target(df_forecast: pd.DataFrame, df_target:pd.DataFrame, on_cols: list[str]):
# #     """
# #     Join target data to forecast data.
        
# #     Args:
# #         df_forecast (pd.DataFrame): DataFrame with forecast data.
        
# #     Returns:
# #         pd.DataFrame: DataFrame with forecast data after joining target data.
# #     """

# #     df_forecast_01 = df_forecast.merge(
# #         df_target,
# #         on=on_cols,
# #         how='outer'
# #     )
    
# #     df_forecast_01['CANT_CLASES'] = df_forecast_01['CANT_CLASES'].fillna(0).astype('int32')
# #     df_forecast_01['CANT_ALUMNOS'] = df_forecast_01['CANT_ALUMNOS'].fillna(0).astype('int32')
# #     df_forecast_01['CANT_ALUMNOS_PREDICT'] = df_forecast_01['CANT_ALUMNOS_PREDICT'].fillna(0).astype('float64')
# #     df_forecast_01['IDX'] = df_forecast_01['IDX'].fillna(0).astype('int32')

# #     return df_forecast_01


# # def calculate_metrics(df_forecast:pd.DataFrame, df_target:pd.DataFrame, on_cols: list[str], logging):
    
# #     df_forecast_02 = join_target(df_forecast, df_target, on_cols)
# #     df_forecast_formado = df_forecast_02[
# #             df_forecast_02['CANT_CLASES'] > 0].copy()
# #     pct_comb = df_forecast_formado['IDX'].mean()
# #     logging.info("% Comb: {:.1%}".format(pct_comb))
# #     # df_forecast_02 = self.calculate_classes(df_forecast_01)
# #     es_zero_predict = df_forecast_02['CANT_CLASES_PREDICT'] == 0
# #     es_zero = df_forecast_02['CANT_CLASES'] == 0
# #     df_forecast_03 = df_forecast_02[~(es_zero_predict & es_zero)].copy()
# #     faltante = np.where(
# #             df_forecast_03['CANT_CLASES'] > df_forecast_03['CANT_CLASES_PREDICT'],
# #             df_forecast_03['CANT_CLASES'] - df_forecast_03['CANT_CLASES_PREDICT'],
# #             0)
# #     sobrante = np.where(
# #             df_forecast_03['CANT_CLASES_PREDICT'] > df_forecast_03['CANT_CLASES'],
# #             df_forecast_03['CANT_CLASES_PREDICT'] - df_forecast_03['CANT_CLASES'],
# #             0)
# #     logging.info("Total Faltante: {:n}".format(faltante.sum()))
# #     logging.info("Total Sobrante: {:n}".format(sobrante.sum()))
# #     logging.info("Total Gestion: {:n}".format(faltante.sum() + sobrante.sum()))
# #     match_clases = np.where(
# #             df_forecast_03['CANT_CLASES'] == df_forecast_03['CANT_CLASES_PREDICT'], 1, 0)
# #     precision = match_clases.mean()
# #     logging.info("% Precision {:.1%}".format(precision))
# #     logging.info("Total Clases (vs Predict): {:,.0f} ({:,.0f})".format(
# #             df_forecast_03['CANT_CLASES'].sum(), df_forecast_03['CANT_CLASES_PREDICT'].sum()))
# #     logging.info("Total alumnos (vs Predict): {:,.0f} ({:,.0f})".format(
# #             df_forecast_03['CANT_ALUMNOS'].sum(), df_forecast_03['CANT_ALUMNOS_PREDICT'].sum()))


# def filter_by_hora_atencion(df_idx: pd.DataFrame, df_turno_disponible: pd.DataFrame, df_horario: pd.DataFrame):
#     df_idx_clone = df_idx.copy()

#     df_idx_clone_01 = df_idx_clone.merge(
#         df_horario[['HORARIO', 'HORA_MIN', 'HORA_MAX']].rename(
#             columns={'HORARIO': 'HORARIO_ACTUAL'}
#         ),
#         on=['HORARIO_ACTUAL'],
#         how='left'
#     )
    
#     assert df_idx_clone_01['HORA_MIN'].isnull().sum() == 0
#     assert df_idx_clone_01['HORA_MAX'].isnull().sum() == 0

#     # df_idx_clone_02 = df_idx_clone_01.copy()
#     df_idx_clone_02 = df_idx_clone_01.rename(
#         columns={'HORA_MIN': 'HORA_INICIO', 'HORA_MAX': 'HORA_FIN'}
#     )

#     # df_idx_clone['HORA_INICIO'] = (
#     #     df_idx_clone['HORARIO_ACTUAL'].str.replace(PATTERN_HORARIO, r"\1", regex=True).astype('int32') +
#     #     df_idx_clone['HORARIO_ACTUAL'].str.replace(PATTERN_HORARIO, r"\2", regex=True).astype('int32')/60)
#     # 
#     # df_idx_clone['HORA_FIN'] = (
#     #     df_idx_clone['HORARIO_ACTUAL'].str.replace(PATTERN_HORARIO, r"\3", regex=True).astype('int32') +
#     #     df_idx_clone['HORARIO_ACTUAL'].str.replace(PATTERN_HORARIO, r"\4", regex=True).astype('int32')/60)
            
#     df_turno_disponible_01 = df_turno_disponible.rename(
#         columns={'PERIODO': 'PERIODO_TARGET'}
#     ).copy()
            
#     df_idx_clone_03 = df_idx_clone_02.merge(
#         df_turno_disponible_01,
#         on=['PERIODO_TARGET', 'SEDE'],
#         how='left'
#     )

#     # if df_idx_clone_01['HORA_MAX'].isnull().sum() == 0:
#     #     pass
#     # else:
#     #     ww = df_idx_clone_01[df_idx_clone_01['HORA_MAX'].isnull()].copy()
#     #     ww.to_excel('output/ww.xlsx', index=False)
#     assert df_idx_clone_03['HORA_MAX'].isnull().sum() == 0

#     filtro = (
#         (df_idx_clone_03['HORA_MAX'] >= df_idx_clone_03['HORA_FIN']) & 
#         (df_idx_clone_03['HORA_MIN'] <= df_idx_clone_03['HORA_INICIO'])
#     )

#     df_idx_clone_04 = df_idx_clone_03[filtro].copy()
    
#     return df_idx_clone_04


    


if __name__ == '__main__':
    x = {'S':[10], 'o':[90, 6]}
    y = list(x.values())
    qq = []
    for isa in y :
        qq.extend(isa)

    print(qq)
    # df = pd.read_parquet('data/platinumdata/paths/prog-acad/2025/202501.parquet')
    # print(df)
