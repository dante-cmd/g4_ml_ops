
"""
Feature Engineering - Inicial Regular
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
import argparse
from pathlib import Path
from unidecode import unidecode
from utils_feats import get_n_lags, get_training_periodos, filter_by_hora_atencion, get_mapping_tipos, parse_args, get_ahead_n_periodos
from parser_regular import Utils
from loader import Loader


SEDES = ['Ica', 'Pucallpa']
SKIPPED_TARGET_PERIODOS = [
    [202503, 202504, 202505, 202506, 202603],
    [202403] + [202501, 202502, 202503]
    ]

SKIPPED_FEATURES_PERIODOS = [
    [202503, 202504, 202505, 202506, 202603],
    [202403, 202404, 202405, 202406] + [202501, 202502, 202503]
    ]
    

class Inicial(Utils):
    def __init__(self, tablas:dict):
        super().__init__(tablas)
        self.df_regular = self.get_regular_plus_synth()
        self.df_real = self.get_regular_plus_synth()
        self.tipo='inicial_regular'
        self.dummy_columns = None

    def get_idx(self, periodo: int):
        meses = [1, 2]
        df_regular = self.df_regular.copy()

        lag_12 = get_n_lags(periodo, 12)
        lag_03 = get_n_lags(periodo, 3)
        lag_02 = get_n_lags(periodo, 2)
        lag_01 = get_n_lags(periodo, 1)

        lag_12_filter = (df_regular['PERIODO'] == lag_12)

        ult_03_filter = ((df_regular['PERIODO'] >= lag_03)
                         & (df_regular['PERIODO'] <= lag_01))

        if periodo % 100 in meses:
            df_regular_inicial = df_regular[
                (lag_12_filter | ult_03_filter)
                & (df_regular['FLAG_INICIAL'] == 1)
                ].copy()

            df_regular_inicial_01 = df_regular_inicial[
                ['SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']].drop_duplicates()
            df_regular_inicial_03 = df_regular_inicial_01.copy()

        elif periodo % 100 == 3:
            df_regular_inicial = df_regular[
                (lag_12_filter | ult_03_filter)
                & (df_regular['FLAG_INICIAL'] == 1)
                ].copy()

            df_regular_inicial_01 = df_regular_inicial[
                ['SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']].drop_duplicates()
            
            df_regular_inicial_02 = df_regular_inicial_01.merge(
                self.df_curso_actual[
                    ['CURSO_ACTUAL', 'NIVEL', 'FRECUENCIA']].copy(),
                on=['CURSO_ACTUAL'],
                how='left')
            assert df_regular_inicial_02['NIVEL'].isnull().sum() == 0

            es_nivel = df_regular_inicial_02['NIVEL'].isin(['ST1', 'ST2', 'SY', 'IST'])
            es_frecuencia = df_regular_inicial_02['FRECUENCIA'].isin(['Interdiario'])
            es_horario = df_regular_inicial_02['HORARIO_ACTUAL'].isin(
                ['07:00 - 08:30', '08:45 - 10:15',
                 '10:30 - 12:00', '12:30 - 14:00',
                 '14:15 - 15:45'])

            df_regular_inicial_03 = df_regular_inicial_02.loc[
                ~(es_nivel & es_frecuencia & es_horario),
                ['SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']
            ].copy()

        else:
            df_regular_inicial = df_regular[
                ult_03_filter
                & (df_regular['FLAG_INICIAL'] == 1)
                ].copy()

            df_regular_inicial_01 = df_regular_inicial[
                ['SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']].drop_duplicates()
            df_regular_inicial_03 = df_regular_inicial_01.copy()

        df_regular_inicial_03['PERIODO_LAG_01'] = lag_01
        df_regular_inicial_03['PERIODO_LAG_02'] = lag_02
        df_regular_inicial_03['PERIODO_LAG_03'] = lag_03
        df_regular_inicial_03['PERIODO_LAG_12'] = lag_12
        df_regular_inicial_03['PERIODO_TARGET'] = periodo
        df_regular_inicial_03['IDX'] = 1

        df_regular_inicial_04 = self.add_pe(
            df_regular_inicial_03, 
            periodo_column='PERIODO_TARGET')
        
        df_regular_inicial_05 = self.add_vac_estandar(
            df_regular_inicial_04, 
            periodo_column='PERIODO_TARGET')

        df_regular_inicial_06 = filter_by_hora_atencion(
                df_regular_inicial_05,
                self.df_turno_disponible,
                self.df_horario
            )

        df_regular_inicial_07 = df_regular_inicial_06[
            ['PERIODO_TARGET', 'PERIODO_LAG_01', 'PERIODO_LAG_02',
             'PERIODO_LAG_03', 'PERIODO_LAG_12', 'SEDE',
             'CURSO_ACTUAL', 'HORARIO_ACTUAL', 'IDX',
             'PE', 'VAC_ACAD_ESTANDAR']].copy()

        return df_regular_inicial_07

    def add_lag_n(self, df_idx: pd.DataFrame, n:int):
        df_regular = self.df_regular[['PERIODO', 'SEDE',
                                      'CURSO_ACTUAL', 'HORARIO_ACTUAL',
                                      'CANT_CLASES', 'CANT_ALUMNOS']].copy()
        assert 12 >= n >= 1
        lag_n = str(n).zfill(2)

        lag_n_df_regular = df_regular.rename(
            columns={'PERIODO':f'PERIODO_LAG_{lag_n}',
                     'CANT_CLASES': f'CANT_CLASES_LAG_{lag_n}',
                     'CANT_ALUMNOS': f'CANT_ALUMNOS_LAG_{lag_n}',
                     }
        )

        df_idx_01 = df_idx.merge(
            lag_n_df_regular,
            on=[f'PERIODO_LAG_{lag_n}', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL'],
            how='left'
        )
        df_idx_01[f'CANT_CLASES_LAG_{lag_n}'] = df_idx_01[f'CANT_CLASES_LAG_{lag_n}'].fillna(0)
        df_idx_01[f'CANT_ALUMNOS_LAG_{lag_n}'] = df_idx_01[f'CANT_ALUMNOS_LAG_{lag_n}'].fillna(0)

        return df_idx_01

    def add_quantitative_feats(self, df_idx:pd.DataFrame):
        df_idx_01 = self.add_lag_n(df_idx, 1)
        df_idx_02 = self.add_lag_n(df_idx_01, 2)
        df_idx_03 = self.add_lag_n(df_idx_02, 3)
        df_idx_12 = self.add_lag_n(df_idx_03, 12)
        return df_idx_12

    def add_categorical_feats(self, df_idx:pd.DataFrame):
        df_idx_01 = df_idx.merge(
            self.df_curso_actual[
                ['CURSO_ACTUAL', 'NIVEL', 'CURSO_2',
                 'FRECUENCIA', 'DURACION']].copy(),
            on=['CURSO_ACTUAL'],
            how='left'
        )

        assert df_idx_01['CURSO_ACTUAL'].isnull().sum() == 0

        return df_idx_01

    def get_target(self, periodo: int):
        df_real = self.df_real.copy()
        df_real_01 = df_real[
            (df_real['PERIODO'] == periodo)
            & (df_real['FLAG_INICIAL'] == 1)].copy()

        df_real_02 = df_real_01[
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL',
             'CANT_CLASES', 'CANT_ALUMNOS']].copy()

        df_real_03 = df_real_02.rename(
            columns={'PERIODO': 'PERIODO_TARGET'}
        )
        
        df_real_03 = df_real_03[
             ~(((df_real_03['SEDE'] == SEDES[0]) & 
               df_real_03['PERIODO_TARGET'].isin(SKIPPED_TARGET_PERIODOS[0]))
               | ((df_real_03['SEDE'] == SEDES[1]) & 
               df_real_03['PERIODO_TARGET'].isin(SKIPPED_TARGET_PERIODOS[1])
               ))
             ].copy()
        
        return df_real_03

    def get_model_by_periodo(self, periodo: int):
        data_idx = self.get_idx(periodo)
        data_target = self.get_target(periodo)

        data_quantitative = self.add_quantitative_feats(data_idx)
        data_model = data_quantitative.merge(
            data_target,
            on=['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL'],
            how='left'
        )

        data_model['CANT_CLASES'] = data_model['CANT_CLASES'].fillna(0)
        data_model['CANT_ALUMNOS'] = data_model['CANT_ALUMNOS'].fillna(0)

        data_model['CANT_CLASES'] = data_model['CANT_CLASES'].astype('int32')
        data_model['CANT_ALUMNOS'] = data_model['CANT_ALUMNOS'].astype('int32')

        return data_model

    def get_model(self, periodos: list[int]):
        collection = []
        for periodo in periodos:
            # print(f'-------TrainContinuidad(periodo={periodo})-------')
            data_model = self.get_model_by_periodo(periodo)
            collection.append(data_model)

        data_model_consol = pd.concat(collection, ignore_index=True)
        data_model_consol_01 = self.add_categorical_feats(data_model_consol)
        data_model_consol_01['LEVEL'] = 'L_' + data_model_consol_01['CURSO_2'].str.replace(
            r'(.+) (.+)', r'\1', regex=True)
        data_model_consol_01['IDX_CURSO'] = 'IC_' + data_model_consol_01['CURSO_2'].str.replace(
            r'(.+) (.+)', r'\2', regex=True)
        return data_model_consol_01

    def get_features(self, periodo: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        periodos = get_training_periodos(periodo)
        data_model = self.get_model(periodos)

        columns_categorical = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']
        
        data_model_01 = data_model.copy()

        # data_model_01['FLAG_PE'] = np.where(data_model_01['CANT_ALUMNOS'] == data_model_01['PE'], 1, 0)
        data_model_01 = data_model_01[
            ~(
                ((data_model_01['SEDE'] == SEDES[0]) & 
                (data_model_01['PERIODO_TARGET'].isin(SKIPPED_FEATURES_PERIODOS[0])))
                | 
                ((data_model_01['SEDE'] == SEDES[1]) & 
                (data_model_01['PERIODO_TARGET'].isin(SKIPPED_FEATURES_PERIODOS[1])))
            )
            ].copy()
        # self.dummy_columns = dummy_columns.copy() + ['FLAG_PE']

        data_model_train = data_model_01[data_model_01.PERIODO_TARGET < periodo].copy()
        data_model_test = data_model_01[data_model_01.PERIODO_TARGET  == periodo].copy()
        
        return (data_model_train, data_model_test)
    
    def load_features(self, 
                      periodo:int, 
                      version:str,
                      output_feats_datastore:str,
                      ):
        # output_feats_train_datastore:str,
        # output_feats_test_datastore:str
        
        data_model_train, data_model_test = self.get_features(periodo)
        
        # ./data/ml_data/features/{self.tipo}/{version}/train/{periodo}/data_feats_{self.tipo}_{periodo}.parquet
        data_model_train.to_parquet(
            f"{output_feats_datastore}/train/{version}/data_feats_{self.tipo}_{periodo}.parquet", 
            index=False)
        data_model_test.to_parquet(
            f"{output_feats_datastore}/test/{version}/data_feats_{self.tipo}_{periodo}.parquet", 
            index=False)
    
    def load_target(self, periodo:int, version:str, output_target_datastore:str):

        try:
            df_real_09 = self.get_target(periodo)
            df_real_09.to_parquet(
                f"{output_target_datastore}/test/{version}/data_target_{self.tipo}_{periodo}.parquet", 
                index=False)
        except Exception as e:
            print(f"Error al cargar target para periodo {periodo}")
            print(e)


def main(args):
    
    # python src/features/inicial_regular.py --input_datastore "./data/ml_data/platinumdata/v1/" --output_feats_datastore "./data/ml_data/features/" --output_target_datastore "./data/ml_data/target/" --feats_version "v1" --target_version "v1" --periodo 202506 --ult_periodo 202506
    # for Loader
    input_datastore=args.input_datastore
    ult_periodo=args.ult_periodo
    
    # for Inicial
    output_feats_inicial_datastore = args.output_feats_datastore
    output_target_inicial_datastore = args.output_target_datastore
    platinum_version = args.platinum_version
    feats_version=args.feats_version
    target_version = args.target_version
    model_periodo=args.model_periodo
    n_eval_periodos=args.n_eval_periodos

    loader = Loader(
        input_datastore, 
        platinum_version,
        ult_periodo)
    
    tablas = loader.fetch_all()

    # Continuidad a nivel de curso
    inicial = Inicial(tablas)
    tipo = inicial.tipo

    # Create directories if they don't exist
    feats_train_path = Path(output_feats_inicial_datastore) / "train" / feats_version
    feats_test_path = Path(output_feats_inicial_datastore) / "test" / feats_version
    target_test_path = Path(output_target_inicial_datastore) / "test" / target_version
    
    feats_train_path.mkdir(parents=True, exist_ok=True)
    feats_test_path.mkdir(parents=True, exist_ok=True)
    target_test_path.mkdir(parents=True, exist_ok=True)

    for periodo in get_ahead_n_periodos(model_periodo, n_eval_periodos):
        mapping_tipos = get_mapping_tipos(periodo)
    
        if mapping_tipos[tipo]:

            inicial.load_features(periodo, feats_version, output_feats_inicial_datastore)
        
            inicial.load_target(periodo, target_version, output_target_inicial_datastore)
        else:
            print(f"No se generaron features para el periodo {periodo} y tipo {tipo}")


if __name__ == '__main__':
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
