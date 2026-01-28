
"""
Feature Engineering - Inicial Estacional
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
import argparse
from pathlib import Path
from unidecode import unidecode
from utils_feats import get_n_lags, get_all_periodos, get_training_periodos_estacionales, filter_by_hora_atencion, parse_args, get_mapping_tipos
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor # type: ignore
from loader import Loader
from parser_estacional import Utils

SEDE = ['Pucallpa']
SKIPPED_PERIODOS =  [202501] # [202401, 202402] +


class Inicial(Utils):
    def __init__(self,tablas:dict):
        super().__init__(tablas)

        self.tipo = 'inicial_estacional'
        self.df_estacional = self.get_estacional_plus_synth()
        self.df_real = self.get_estacional()
        self.dummy_columns = None

    def get_idx(self, periodo: int):
        df_estacional = self.df_estacional.copy()
        
        lag_12 = get_n_lags(periodo, 12)
        lag_01 = get_n_lags(periodo, 1)

        if periodo % 100 == 1:
            df_estacional_inicial = df_estacional[
                    (df_estacional['PERIODO'].isin([lag_12]))
                    & (df_estacional['FLAG_INICIAL'] == 1)
            ].copy()
            df_estacional_inicial['PERIODO_TARGET'] = periodo
            df_estacional_inicial['PERIODO_LAG_12'] = lag_12
            df_estacional_inicial['IDX'] = 1

            df_estacional_inicial_01 = filter_by_hora_atencion(
                df_estacional_inicial,
                self.df_turno_disponible,
                self.df_horario
            )

            df_estacional_inicial_02 = df_estacional_inicial_01[
                ['PERIODO_TARGET', 'PERIODO_LAG_12', 'SEDE',
                 'CURSO_ACTUAL', 'HORARIO_ACTUAL', 'IDX',
                 'PE', 'VAC_ACAD_ESTANDAR']].copy()


            return df_estacional_inicial_02
        else:
            df_estacional_inicial = df_estacional[
                    (df_estacional['PERIODO'].isin([lag_01]))
                    & (df_estacional['FLAG_INICIAL'] == 1)
            ].copy()
            
            df_estacional_inicial['PERIODO_TARGET'] = periodo
            df_estacional_inicial['PERIODO_LAG_01'] = lag_01
            df_estacional_inicial['IDX'] = 1

            df_estacional_inicial_01 = filter_by_hora_atencion(
                df_estacional_inicial,
                self.df_turno_disponible,
                self.df_horario
            )

            df_estacional_inicial_02 = df_estacional_inicial_01[
                ['PERIODO_TARGET', 'PERIODO_LAG_01', 'SEDE',
                'CURSO_ACTUAL', 'HORARIO_ACTUAL', 'IDX',
                'PE', 'VAC_ACAD_ESTANDAR']].copy()

            return df_estacional_inicial_02

    def add_lag_n(self, df_idx: pd.DataFrame, n:int):
        assert 12 >= n >= 1
        lag_n = str(n).zfill(2)

        df_estacional = self.df_estacional[['PERIODO', 'SEDE',
                                      'CURSO_ACTUAL', 'HORARIO_ACTUAL',
                                      'CANT_CLASES', 'CANT_ALUMNOS']].copy()

        df_estacional_01 = df_estacional.rename(
            columns={'PERIODO':f'PERIODO_LAG_{lag_n}',
                     'CANT_CLASES': f'CANT_CLASES_LAG_{lag_n}',
                     'CANT_ALUMNOS': f'CANT_ALUMNOS_LAG_{lag_n}',
                     }
        )

        df_idx_01 = df_idx.merge(
            df_estacional_01,
            on=[f'PERIODO_LAG_{lag_n}', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL'],
            how='left'
        )
        df_idx_01[f'CANT_CLASES_LAG_{lag_n}'] = df_idx_01[f'CANT_CLASES_LAG_{lag_n}'].fillna(0)
        df_idx_01[f'CANT_ALUMNOS_LAG_{lag_n}'] = df_idx_01[f'CANT_ALUMNOS_LAG_{lag_n}'].fillna(0)

        return df_idx_01

    def add_quantitative_feats(self, periodo:int, df_idx:pd.DataFrame):
        if periodo % 100 == 1:
            df_idx_01 = self.add_lag_n(df_idx, 12)
            return df_idx_01
        else:
            df_idx_01 = self.add_lag_n(df_idx, 1)
            return df_idx_01

    def add_categorical_feats(self, df_idx):
        df_idx_01 = df_idx.merge(
            self.df_curso_actual[['CURSO_ACTUAL', 'NIVEL', 'CURSO_2']].copy(),
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
                columns={'PERIODO':'PERIODO_TARGET'}
            )
        df_real_03 = df_real_03[
            ~((df_real_03['SEDE'].isin(SEDE)) &
              (df_real_03['PERIODO_TARGET'].isin(SKIPPED_PERIODOS)))
              ].copy()
        
        return df_real_03

    def get_model_by_periodo(self, periodo: int):
        data_idx = self.get_idx(periodo)
        data_target = self.get_target(periodo)

        data_quantitative = self.add_quantitative_feats(periodo, data_idx)
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
        # meses = [1, 2]
        if periodo % 100 in [1]:
            periodos = get_training_periodos_estacionales(periodo, [1])
        elif periodo % 100 in [2]:
            periodos = get_training_periodos_estacionales(periodo, [2])
        else:
            return (pd.DataFrame(), pd.DataFrame())

        data_model = self.get_model(periodos)

        columns_categorical = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']
        
        data_model_01 = data_model.copy()

        data_model_01 = data_model_01[
            ~((data_model_01['SEDE'].isin(SEDE)) &
              (data_model_01['PERIODO_TARGET'].isin(SKIPPED_PERIODOS)))
              ].copy()
        
        # data_model_01['FLAG_PE'] = np.where(data_model_01['CANT_ALUMNOS'] == data_model_01['PE'], 1, 0)
        # self.dummy_columns = dummy_columns.copy() + ['FLAG_PE']

        data_model_train = data_model_01[data_model_01.PERIODO_TARGET < periodo].copy()
        data_model_test = data_model_01[data_model_01.PERIODO_TARGET  == periodo].copy()

        return (data_model_train, data_model_test)

    def load_features(self, periodo:int, version:str, output_feats_datastore:str):
        
        data_model_train, data_model_test = self.get_features(periodo)
        data_model_train.to_parquet(
            f"{output_feats_datastore}/{self.tipo}/train/{version}/data_feats_{self.tipo}_{periodo}.parquet", index=False)
        data_model_test.to_parquet(
            f"{output_feats_datastore}/{self.tipo}/test/{version}/data_feats_{self.tipo}_{periodo}.parquet", index=False)
    
    def load_target(self, periodo:int, version:str, output_target_datastore:str):

        try:
            df_real_09 = self.get_target(periodo)
            df_real_09.to_parquet(
                f"{output_target_datastore}/{self.tipo}/test/{version}/data_target_{self.tipo}_{periodo}.parquet", index=False)
        except Exception as e:
            print(f"Error al cargar target para periodo {periodo}")
            print(e)

def main(args):
    input_datastore=args.input_datastore
    ult_periodo=args.ult_periodo
    
    # for Inicial
    output_feats_datastore = args.output_feats_datastore
    output_target_datastore = args.output_target_datastore
    platinum_version = args.platinum_version
    feats_version=args.feats_version
    target_version=args.target_version
    periodo=args.periodo
    
    
    loader = Loader(
        input_datastore, 
        platinum_version,
        ult_periodo)
    
    tablas = loader.fetch_all()

    inicial = Inicial(tablas)
    
    # Create directories if they don't exist
    tipo = inicial.tipo

    mapping_tipos = get_mapping_tipos(periodo)

    if mapping_tipos[tipo]:
        feats_train_path = Path(output_feats_datastore) / tipo / "train" / feats_version
        feats_test_path = Path(output_feats_datastore) / tipo / "test" / feats_version
        target_test_path = Path(output_target_datastore) / tipo / "test" / target_version
        
        feats_train_path.mkdir(parents=True, exist_ok=True)
        feats_test_path.mkdir(parents=True, exist_ok=True)
        target_test_path.mkdir(parents=True, exist_ok=True)

        inicial.load_features(
            periodo, 
            feats_version,
            output_feats_datastore)

        inicial.load_target(
            periodo, 
            target_version,
            output_target_datastore)
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
    print("\n\n")