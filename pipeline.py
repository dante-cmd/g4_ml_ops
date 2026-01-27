from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import pandas as pd
from pathlib import Path
from unidecode import unidecode
from datetime import datetime
from dateutil.relativedelta import relativedelta
# ------------------------------------------------------------------------

from src.data.etl import Etl
from features.loader import Loader as DataTrain
from features.continuidad_estacional import Inicial as InicialEstacional
from features.continuidad_estacional import Continuidad as ContinuidadEstacional
from features.continuidad_regular import Inicial as InicialRegular
from features.continuidad_regular import Continuidad as ContinuidadRegular
from features.continuidad_regular import ContinuidadToHorario as ContinuidadRegularToHorario
from models.inicial_estacional import TrainInicial as TrainInicialEstacional
from models.inicial_estacional import TrainContinuidad as TrainContinuidadEstacional
from models.continuidad_regular import TrainInicial as TrainInicialRegular
from models.continuidad_regular import TrainContinuidad as TrainContinuidadRegular
from models.continuidad_regular import TrainContinuidadToHorario as TrainContinuidadRegularToHorario
from utils import calculate_classes, calculate_metrics, join_target,  save_output
from utils_azure import get_path_to_read, data_uploader
import logging


logging.basicConfig(
        filename='logging/app.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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


def exec_job_etl():
    
    # --------------------------------------------------------------------
    # ---------------------------- ETL -----------------------------------
    # --------------------------------------------------------------------

    ult_periodo = 202512
    n_periodos = "None"
    all_periodos = "True"
    input_folder_root = input_datastore = 'rawdata'
    output_folder_root = output_datastore = 'platinumdata'
    compute = ''
    display_name = ''
    experiment_name = ''

    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    job_inputs = {
        "datastore_data": Input(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{input_folder_root}")}

    job_outputs = {
        "datastore_data": Output(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{output_datastore}/paths/{output_folder_root}")
    }
    
    # configure job
    job = command(
        code="./src",
        command="python -m data.etl \
            --input_datastore ${{inputs.datastore_data}} \
            --output_datastore ${{outputs.datastore_data}}\
            --ult_periodo %d\
            --n_periodos '%s'\
            --all_periodos '%s'"% (ult_periodo, n_periodos, all_periodos),
        inputs=job_inputs,
        outputs=job_outputs,
        environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
        compute=compute,
        display_name=display_name,
        experiment_name=experiment_name
    )

    # submit job
    returned_job = ml_client.create_or_update(job)
   

def exec_job_models():
    pass


def exec_job_loader():
     
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    job_inputs = {
        "datastore_data": Input(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{input_datastore}/paths/{input_folder_root}")}

    job_outputs = {
        "datastore_data": Output(
            type=AssetTypes.URI_FOLDER, 
            path=f"azureml://datastores/{output_datastore}/paths/LocalUpload/{output_folder_root}")
    }
    
    # configure job
    job = command(
        code="./src",
        command="python -m data.etl \
            --input_datastore ${{inputs.datastore_data}} \
            --output_datastore ${{outputs.datastore_data}}\
            --ult_periodo %d\
            --n_periodos '%s'\
            --all_periodos '%s'"% (ult_periodo, n_periodos, all_periodos),
        inputs=job_inputs,
        outputs=job_outputs,
        environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
        compute=compute,
        display_name=display_name,
        experiment_name=experiment_name
    )

    # submit job
    returned_job = ml_client.create_or_update(job)
    pass

    
def pipeline():
    # --------------------------------------------------------------------
    # --------------------------- Parameters -----------------------------
    # --------------------------------------------------------------------
    
    base_path = 'data'
    where = 'cloud'
    raw_data_asset_path = get_path(where, base_path, 'rawdata')
    platinum_data_asset_path = get_path(where, base_path, 'platinum')
    periodo = 202512
    ult_periodo = 202512
    n_periodos = None
    all_periodos = True
    
    output = True
    mode = 'dev'
    columns = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL',
                'CANT_CLASES_PREDICT', 'CANT_ALUMNOS_PREDICT']
    

    # --------------------------------------------------------------------
    # ---------------------------- ETL -----------------------------------
    # --------------------------------------------------------------------

    data_etl = Etl(raw_data_asset_path, 
                   platinum_data_asset_path,
                   ult_periodo,
                   n_periodos,
                   all_periodos)

    data_etl.fetch_and_transform_and_pull()

    # --------------------------------------------------------------------
    # --------------------------- Data Train -----------------------------
    # --------------------------------------------------------------------
    # **Get all data needed for the models**
    # 1. tabla_curso_actual
    # 2. tabla_curso_acumulado
    # 3. tabla_curso_inicial
    # 4. tabla_pe
    # 5. tabla_vac_estandar
    # 6. tabla_horario_diario_to_sabatino
    # 7. tabla_curso_diario_to_sabatino
    # 8. prog_acad
    # 9. synthetic

    data_train = DataTrain(base_path, raw_data_store, platinum_data_store,
                           ult_periodo, n_periodos, all_periodos, credential)
    tablas = data_train.fetch_all()

    tablas['tabla_curso_actual']
    tablas['tabla_curso_acumulado']
    tablas['tabla_curso_inicial']
    tablas['tabla_pe']
    tablas['tabla_horario']
    tablas['tabla_vac_estandar']
    tablas['tabla_horario_diario_to_sabatino']
    tablas['tabla_curso_diario_to_sabatino']
    tablas['prog_acad']
    tablas['synthetic']
    tablas['tabla_turno_disponible']

    estacional_inicial, estacional_continuidad, regular_inicial, regular_continuidad = get_tipo_train(periodo)
    
    
    assert mode in ['dev', 'prod']
    
    logging.info(f' {mode} {periodo} '.center(50, '=').upper())
    

    if estacional_inicial:
        assert periodo%100 in [1,2], "Periodo no es estacional inicial"
        # --------------------------------------------------------------------
        # ------------------------ Inicial Estacional ------------------------
        # --------------------------------------------------------------------

        # ** Features **
        data_inicial_estacional = InicialEstacional(
            prog_acad,
            synthetic,
            tabla_curso_actual,
            tabla_curso_acumulado,
            tabla_curso_inicial,
            tabla_vac_estandar,
            tabla_pe,
            tabla_turno_disponible,
            tabla_horario)

        df_feats_inicial_estacional = data_inicial_estacional.get_features(periodo)
        df_target_inicial_estacional = data_inicial_estacional.get_target(periodo)
        
        # ** Training **
        train_inicial_estacional = TrainInicialEstacional(
            df_target_inicial_estacional,
            df_feats_inicial_estacional,
            periodo,
            data_inicial_estacional.dummy_columns,
            logging
        )

        df_forecast_inicial_estacional = train_inicial_estacional.training()

        # ** Metrics and prediction**
        train_inicial_estacional.metrics(df_forecast_inicial_estacional, mode)
        
        if output:
            df_forecast_inicial_estacional = calculate_classes(df_forecast_inicial_estacional)
            if mode=='dev':
                df_forecast_inicial_estacional = join_target(
                    df_forecast_inicial_estacional,
                    df_target_inicial_estacional,
                    ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']
                    )
                save_output(
                    mode,
                    periodo,
                    df_forecast_inicial_estacional,
                    'forecast_inicial_estacional'
                )
                # df_forecast_inicial_estacional.to_excel(
                #     'output/forecast_inicial_estacional_{}_{}.xlsx'.format(periodo, mode), 
                # index=False)
            else:
                df_forecast_inicial_estacional = df_forecast_inicial_estacional.loc[
                    df_forecast_inicial_estacional['CANT_CLASES_PREDICT']>0, columns].copy()
                save_output(
                    mode,
                    periodo,
                    df_forecast_inicial_estacional,
                    'forecast_inicial_estacional'
                )
                
                # df_forecast_inicial_estacional.to_excel(
                #     'output/forecast_inicial_estacional_{}_{}.xlsx'.format(periodo, mode), 
                # index=False)

    if estacional_continuidad:
        assert periodo%100 in [1,2], "Periodo no es estacional inicial"
        # --------------------------------------------------------------------
        # --------------------- Continuidad Estacional -----------------------
        # --------------------------------------------------------------------

        # ** Features and target **
        data_continuidad_estacional = ContinuidadEstacional(
            prog_acad,
            synthetic,
            tabla_curso_actual,
            tabla_curso_acumulado,
            tabla_curso_inicial,
            tabla_vac_estandar,
            tabla_pe,
            tabla_turno_disponible,
            tabla_horario)

        df_feats_continuidad_estacional = data_continuidad_estacional.get_features(periodo)
        df_target_continuidad_estacional = data_continuidad_estacional.get_target(periodo)

        # ** Training **
        train_continuidad_estacional = TrainContinuidadEstacional(
            df_target_continuidad_estacional,
            df_feats_continuidad_estacional,
            periodo,
            data_continuidad_estacional.dummy_columns,
            logging
        )

        df_forecast_continuidad_estacional = train_continuidad_estacional.training()

        # ** Metrics and prediction**
        train_continuidad_estacional.metrics(
            df_forecast_continuidad_estacional,
            mode)
        
        if output:
            df_forecast_continuidad_estacional = calculate_classes(df_forecast_continuidad_estacional)
            if mode=='dev':
                df_forecast_continuidad_estacional = join_target(
                    df_forecast_continuidad_estacional,
                    df_target_continuidad_estacional,
                    ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']
                    )
                save_output(
                    mode,
                    periodo,
                    df_forecast_continuidad_estacional,
                    'forecast_continuidad_estacional'
                )
                # df_forecast_continuidad_estacional.to_excel(
                # 'output/forecast_continuidad_estacional_{}_{}.xlsx'.format(periodo, mode), index=False)
            else:    
                df_forecast_continuidad_estacional = df_forecast_continuidad_estacional.loc[
                    df_forecast_continuidad_estacional['CANT_CLASES_PREDICT']>0, columns].copy()
                
                save_output(
                    mode,
                    periodo,
                    df_forecast_continuidad_estacional,
                    'forecast_continuidad_estacional'
                )
                
                # df_forecast_continuidad_estacional.to_excel(
                #     'output/forecast_continuidad_estacional_{}_{}.xlsx'.format(periodo, mode), index=False)

    if regular_inicial:
        # --------------------------------------------------------------------
        # --------------------- Inicial Regular --------------------------
        # --------------------------------------------------------------------
        # ** Features **
        data_inicial_regular = InicialRegular(
            prog_acad,
            synthetic,
            tabla_curso_actual,
            tabla_curso_acumulado,
            tabla_curso_inicial,
            tabla_vac_estandar,
            tabla_pe,
            tabla_horario_diario_to_sabatino,
            tabla_curso_diario_to_sabatino,
            tabla_turno_disponible,
            tabla_horario)

        df_features_inicial_regular = data_inicial_regular.get_features(periodo)
        df_target_inicial_regular = data_inicial_regular.get_target(periodo)
        # data_features_inicial_regular.to_excel('output/data_features_inicial_regular.xlsx', index=False)
        # data_target_inicial_regular.to_excel('data_target_inicial_regular.xlsx', index=False)

        # ** Training **
        train_inicial_regular = TrainInicialRegular(
            df_target_inicial_regular,
            df_features_inicial_regular,
            periodo,
            data_inicial_regular.dummy_columns,
            logging
        )

        df_forecast_inicial_regular = train_inicial_regular.training()

        # ** Metrics and prediction**
        train_inicial_regular.metrics(df_forecast_inicial_regular, mode)

        if output:
            df_forecast_inicial_regular = calculate_classes(df_forecast_inicial_regular)
            if mode=='dev':
                df_forecast_inicial_regular = join_target(
                    df_forecast_inicial_regular,
                    df_target_inicial_regular,
                    ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']
                    )
                
                save_output(
                    mode,
                    periodo,
                    df_forecast_inicial_regular,
                    'forecast_inicial_regular'
                )
                # df_forecast_inicial_regular.to_excel(
                # 'output/forecast_inicial_regular_{}_{}.xlsx'.format(periodo, mode), 
                # index=False)
            
            else:
                df_forecast_inicial_regular = df_forecast_inicial_regular.loc[
                    df_forecast_inicial_regular['CANT_CLASES_PREDICT']>0, columns].copy()
                
                save_output(
                    mode,
                    periodo,
                    df_forecast_inicial_regular,
                    'forecast_inicial_regular'
                )
                
                # df_forecast_inicial_regular.to_excel(
                #     'output/forecast_inicial_regular_{}_{}.xlsx'.format(periodo, mode), 
                # index=False)
    
    if regular_continuidad:
        # --------------------------------------------------------------------
        # --------------------- Continuidad Regular --------------------------
        # --------------------------------------------------------------------
        # ** Features **
        data_continuidad_regular = ContinuidadRegular(
            prog_acad,
            synthetic,
            tabla_curso_actual,
            tabla_curso_acumulado,
            tabla_curso_inicial,
            tabla_vac_estandar,
            tabla_pe,
            tabla_horario_diario_to_sabatino,
            tabla_curso_diario_to_sabatino,
            tabla_turno_disponible,
            tabla_horario)

        df_features_continuidad_regular = data_continuidad_regular.get_features(periodo)
        df_target_continuidad_regular = data_continuidad_regular.get_target(periodo)

        # df_features_continuidad_regular.to_excel(
        #     'data_features_continuidad_regular.xlsx', index=False)
        # df_target_continuidad_regular.to_excel(
        #     'data_target_continuidad_regular.xlsx', index=False)

        train_continuidad_regular = TrainContinuidadRegular(
            df_target_continuidad_regular,
            df_features_continuidad_regular,
            periodo,
            data_continuidad_regular.dummy_columns,
            logging
        )
        
        df_forecast_continuidad_regular = train_continuidad_regular.training()

        train_continuidad_regular.metrics(df_forecast_continuidad_regular, mode)
        # forecast_continuidad_regular = join_target(
        #     df_forecast_continuidad_regular)
        # forecast_continuidad_regular = calculate_classes(
        #     forecast_continuidad_regular)
        # forecast_continuidad_regular.to_excel('forecast_continuidad_regular.xlsx', index=False)

        # ------------------------------------------------------------------------------

        data_continuidad_regular_to_horario = ContinuidadRegularToHorario(
            prog_acad,
            synthetic,
            tabla_curso_actual,
            tabla_curso_acumulado,
            tabla_curso_inicial,
            tabla_vac_estandar,
            tabla_pe,
            tabla_horario_diario_to_sabatino,
            tabla_curso_diario_to_sabatino,
            tabla_turno_disponible,
            tabla_horario)

        df_features_continuidad_regular_to_horario = data_continuidad_regular_to_horario.get_features(
            periodo)
        df_target_continuidad_regular_to_horario = data_continuidad_regular_to_horario.get_target(
            periodo)

        # df_features_continuidad_regular_to_horario.to_excel(
        #     'data_features_continuidad_regular_to_horario.xlsx', index=False)
        # df_target_continuidad_regular_to_horario.to_excel(
        #     'data_target_continuidad_regular_to_horario.xlsx', index=False)

        train_continuidad_regular_to_horario = TrainContinuidadRegularToHorario(
            df_target_continuidad_regular_to_horario,
            df_features_continuidad_regular_to_horario,
            df_forecast_continuidad_regular,
            periodo,
            data_continuidad_regular_to_horario.dummy_columns,
            logging
        )
        
        df_forecast_continuidad_regular_to_horario = (
            train_continuidad_regular_to_horario.training())

        train_continuidad_regular_to_horario.metrics(
            df_forecast_continuidad_regular_to_horario, mode)
        
        df_forecast_continuidad_regular_to_horario = train_continuidad_regular_to_horario.fac_to_cant(
            df_forecast_continuidad_regular_to_horario)

        if output:
            df_forecast_continuidad_regular_to_horario = calculate_classes(
                df_forecast_continuidad_regular_to_horario)
            if mode=='dev':
                df_forecast_continuidad_regular_to_horario = join_target(
                    df_forecast_continuidad_regular_to_horario,
                    df_target_continuidad_regular_to_horario,
                    ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']
                    )
                
                save_output(
                    mode,
                    periodo,
                    df_forecast_continuidad_regular_to_horario,
                    'forecast_continuidad_regular_to_horario'
                )
                # df_forecast_continuidad_regular_to_horario.to_excel(
                # 'output/forecast_continuidad_regular_to_horario_{}_{}.xlsx'.format(periodo, mode), 
                # index=False)
                
            else:
                df_forecast_continuidad_regular_to_horario = df_forecast_continuidad_regular_to_horario.loc[
                    df_forecast_continuidad_regular_to_horario['CANT_CLASES_PREDICT']>0, columns].copy()
                
                save_output(
                    mode,
                    periodo,
                    df_forecast_continuidad_regular_to_horario,
                    'forecast_continuidad_regular_to_horario'
                )
                # df_forecast_continuidad_regular_to_horario.to_excel(
                # 'output/forecast_continuidad_regular_to_horario_{}_{}.xlsx'.format(periodo, mode), 
                # index=False)
        
    logging.info(' END '.center(50, '='))


if __name__ == '__main__':
    run_etl()
    pass
