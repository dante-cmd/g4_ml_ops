
"""
Feature Engineering - Continuidad Estacional
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
from utils_feats import get_n_lags, get_all_periodos, get_training_periodos_estacionales, filter_by_hora_atencion, parse_args, get_mapping_tipos
from loader import Loader
from parser_estacional import Utils


# SEDE = ['Pucallpa']
# SKIPPED_PERIODOS = [202401, 202402]

class Continuidad(Utils):
    def __init__(self, tablas:dict):
        super().__init__(tablas)

        self.tipo = 'continuidad_estacional'
        self.df_estacional = self.get_estacional_plus_synth()
        self.df_real = self.get_estacional()
        self.dummy_columns = None

    def get_idx(self, periodo: int):
        df_estacional = self.df_estacional[
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL',
             'PE', 'POSTERIOR_(+1)', 'FLAG_INICIAL', 
             'CANT_CLASES', 'CANT_ALUMNOS']].copy()
        # lag_12(periodo)
        lag_01_periodo = get_n_lags(periodo, 1)

        df_estacional_inicial = df_estacional[
            (df_estacional['PERIODO'] == lag_01_periodo)
            & (df_estacional['FLAG_INICIAL'] == 1)
        ].copy()

        df_estacional_inicial['PERIODO_TARGET'] = periodo

        df_estacional_inicial['IDX'] = 1

        df_estacional_inicial_01 = df_estacional_inicial.rename(
            columns={'CURSO_ACTUAL': 'CURSO_ANTERIOR',
                     'PE':'PE_ANTERIOR'}
        )

        df_estacional_inicial_02 = df_estacional_inicial_01.rename(
            columns={'POSTERIOR_(+1)': 'CURSO_ACTUAL'}
        )

        es_null = df_estacional_inicial_02['CURSO_ACTUAL'].isnull()
        es_fin_icpna = df_estacional_inicial_02['CURSO_ACTUAL'] == 'Fin Icpna'

        df_estacional_inicial_03 = df_estacional_inicial_02[
            ~(es_fin_icpna|es_null)].copy()

        df_estacional_inicial_04 = df_estacional_inicial_03.merge(
            self.df_curso_actual[['CURSO_ACTUAL', 'PROGRAMA',
                                  'NIVEL', 'LINEA_DE_NEGOCIO', 'CURSO_2']].copy(),
            on=['CURSO_ACTUAL'],
            how='left'
        )

        assert df_estacional_inicial_04['PROGRAMA'].isnull().sum() == 0

        es_curso_inicial = df_estacional_inicial_04['CURSO_2'].isin(
            self.df_curso_inicial['CURSOS_INICIALES'])
        
        df_estacional_inicial_05 = df_estacional_inicial_04[
            ~es_curso_inicial].copy()

        df_estacional_inicial_05['PROGRAMA'] = np.where(
            df_estacional_inicial_05['PROGRAMA'] == 'Niños',
            'Niños',
            'Adultos'
        )

        df_vac_estandar = self.df_vac_estandar.rename(
            columns={'PERIODO':'PERIODO_TARGET'}
        ).copy()

        df_pe = self.df_pe.rename(
            columns={'PERIODO': 'PERIODO_TARGET'}
        ).copy()

        df_estacional_inicial_06 = df_estacional_inicial_05.merge(
            df_vac_estandar,
            on=['PERIODO_TARGET', 'LINEA_DE_NEGOCIO', 'NIVEL'],
            how='left'
        )
        
        assert df_estacional_inicial_06['VAC_ACAD_ESTANDAR'].isnull().sum() == 0

        df_estacional_inicial_07 = df_estacional_inicial_06.merge(
            df_pe,
            on=['PERIODO_TARGET', 'PROGRAMA'],
            how='left'
        )

        assert df_estacional_inicial_07['PE'].isnull().sum() == 0

        df_estacional_inicial_08 = filter_by_hora_atencion(
                df_estacional_inicial_07,
                self.df_turno_disponible,
                self.df_horario
            )

        df_estacional_inicial_09 = df_estacional_inicial_08[
            ['PERIODO_TARGET', 'PERIODO', 'SEDE',
             'CURSO_ANTERIOR', 'CURSO_ACTUAL', 'HORARIO_ACTUAL', 'IDX',
             'PE_ANTERIOR', 'PE', 'VAC_ACAD_ESTANDAR']].copy()

        return df_estacional_inicial_09

    def add_anterior(self, df_idx: pd.DataFrame):
        df_estacional = self.df_estacional[['PERIODO', 'SEDE',
                                            'CURSO_ACTUAL', 'HORARIO_ACTUAL',
                                            'CANT_CLASES', 'CANT_ALUMNOS']].copy()
        df_estacional_01 = df_estacional.rename(
            columns={'CURSO_ACTUAL':'CURSO_ANTERIOR'})

        df_idx_01 = df_idx.merge(
            df_estacional_01,
            on=['PERIODO', 'SEDE', 'CURSO_ANTERIOR', 'HORARIO_ACTUAL'],
            how='left'
        )
        df_idx_01['CANT_CLASES'] = df_idx_01['CANT_CLASES'].fillna(0)
        df_idx_01['CANT_ALUMNOS'] = df_idx_01['CANT_ALUMNOS'].fillna(0)
        df_idx_02 = df_idx_01.rename(
            columns={'CANT_CLASES':'CANT_CLASES_ANTERIOR',
                     'CANT_ALUMNOS': 'CANT_ALUMNOS_ANTERIOR',
                     }
        )
        return df_idx_02

    def add_quantitative_feats(self, df_idx):
        df_idx_01 = self.add_anterior(df_idx)
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
            & (df_real['FLAG_INICIAL'] == 0)].copy()

        df_real_02 = df_real_01[
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL',
             'CANT_CLASES', 'CANT_ALUMNOS']].copy()

        df_real_03 = df_real_02.rename(
                columns={'PERIODO':'PERIODO_TARGET'}
            )
        # df_real_03 = df_real_03[
        #     ~((df_real_03['SEDE'].isin(SEDE)) &
        #       (df_real_03['PERIODO_TARGET'].isin(SKIPPED_PERIODOS))
        #       )].copy()
        
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

        data_model['CANT_CLASES'] = data_model['CANT_CLASES'].fillna(0).astype('int32')
        data_model['CANT_ALUMNOS'] = data_model['CANT_ALUMNOS'].fillna(0).astype('int32')

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

    def get_features(self, periodo: int)-> tuple[pd.DataFrame, pd.DataFrame]:
        meses = [2]
        if periodo % 100 not in meses:
            return (pd.DataFrame(), pd.DataFrame())

        periodos = get_training_periodos_estacionales(periodo, meses)
        data_model = self.get_model(periodos)

        columns_categorical = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']

        data_model_01 = data_model.copy()
        
        # data_model_01 = data_model_01[
        #     ~((data_model_01['SEDE'].isin(SEDE)) &
        #       (data_model_01['PERIODO_TARGET'].isin(SKIPPED_PERIODOS))
        #       )].copy()
        
        # data_model_01['FLAG_PE'] = np.where(
        #     data_model_01['CANT_ALUMNOS_ANTERIOR'] == data_model_01['PE_ANTERIOR'],
        #     1, 0)
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
    
    # for Continuidad
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

    # Continuidad a nivel de curso
    continuidad = Continuidad(tablas)
    
    # Create directories if they don't exist
    tipo = continuidad.tipo

    mapping_tipos = get_mapping_tipos(periodo)
    
    if mapping_tipos[tipo]:
        feats_train_path = Path(output_feats_datastore) / tipo / "train" / feats_version
        feats_test_path = Path(output_feats_datastore) / tipo / "test" / feats_version
        target_test_path = Path(output_target_datastore) / tipo / "test" / target_version
        
        feats_train_path.mkdir(parents=True, exist_ok=True)
        feats_test_path.mkdir(parents=True, exist_ok=True)
        target_test_path.mkdir(parents=True, exist_ok=True)
    
        continuidad.load_features(periodo, feats_version, output_feats_datastore)
        continuidad.load_target(periodo, target_version, output_target_datastore)
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
