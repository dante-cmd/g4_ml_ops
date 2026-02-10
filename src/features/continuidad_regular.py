
"""
Feature Engineering - Continuidad Regular
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
from pathlib import Path
from unidecode import unidecode
import argparse
from utils_feats import get_n_lags, get_training_periodos, filter_by_hora_atencion, parse_args, get_mapping_tipos, get_ahead_n_periodos
from parser_regular import Utils
from loader import Loader


SEDES = ['Ica', 'Pucallpa']


SKIPPED_TARGET_PERIODOS = [
    [202503, 202504, 202505] + [202603],
    [202403] + [202501, 202502, 202503]
    ]


SKIPPED_FEATURES_PERIODOS = [
    [202503, 202504, 202505] + [202603],
    [202403, 202404, 202405] + [202501, 202502, 202503]
    ]
# 202407


class Continuidad(Utils):
    """
    Clase para el cálculo de features de continuidad regular.
    """
    def __init__(self, tablas:dict):
        super().__init__(tablas)
        self.tipo = 'continuidad_regular'
        self.df_real = self.get_regular_plus_synth()
        self.df_regular = self.get_regular_plus_synth()
        self.df_regular_full = self.get_regular_plus_synth_plus_estacional()
        self.dummy_columns = None

    def get_idx(self, periodo: int):
        """
        Obtiene el índice de continuidad regular para un periodo dado.
        """
        lag_01 = get_n_lags(periodo, 1)
        lag_02 = get_n_lags(periodo, 2)
        lag_12 = get_n_lags(periodo, 12)
        
        df_regular_full = self.df_regular_full.copy()

        df_regular_full = df_regular_full.merge(
            self.df_curso_actual[
                ['CURSO_ACTUAL', 'NIVEL', 'FRECUENCIA',
                 'DURACION', 'POSTERIOR_(+1)']],
            on=['CURSO_ACTUAL'],
            how='left'
        )

        assert df_regular_full['NIVEL'].isnull().sum() == 0

        es_base = (((df_regular_full['PERIODO'] == lag_01)
                    & (df_regular_full['DURACION'] == "Mensual")) |
                   ((df_regular_full['PERIODO'] == lag_02)
                    & (df_regular_full['DURACION'] == "Bimensual")))

        es_estacional = (df_regular_full['FLAG_ESTACIONAL'] == 1)

        es_fin = ((df_regular_full['POSTERIOR_(+1)'] == 'Fin Icpna') |
                  df_regular_full['POSTERIOR_(+1)'].isnull())

        es_nivel = df_regular_full['NIVEL'].isin(['ST1', 'ST2', 'SY', 'IST'])
        es_frecuencia = df_regular_full['FRECUENCIA'].isin(['Interdiario'])
        es_horario = df_regular_full['HORARIO_ACTUAL'].isin(
            ['07:00 - 08:30', '08:45 - 10:15', '10:30 - 12:00',
             '12:30 - 14:00', '14:15 - 15:45'])

        if periodo % 100 in [3]:

            df_regular_continuidad_estacional = df_regular_full[
                es_base & es_estacional].copy()

            df_regular_continuidad_estacional_01 = df_regular_continuidad_estacional.drop(
                columns=['POSTERIOR_(+1)'])

            df_regular_continuidad_estacional_02 = df_regular_continuidad_estacional_01.merge(
                self.df_curso_diario_to_sabatino[['CURSO_ACTUAL', 'POSTERIOR_(+1)']].copy(),
                how='left',
                on=['CURSO_ACTUAL']
            )

            assert df_regular_continuidad_estacional_02['POSTERIOR_(+1)'].isnull().sum() == 0

            df_regular_continuidad_estacional_03 = df_regular_continuidad_estacional_02[
                ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'POSTERIOR_(+1)']].drop_duplicates()

            df_regular_continuidad = df_regular_full[
                es_base & ~es_estacional & ~es_fin &
                ~(es_nivel & es_frecuencia & es_horario)].copy()

            df_regular_continuidad_01 = df_regular_continuidad[
                ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'POSTERIOR_(+1)']].drop_duplicates()

            df_regular_continuidad_full = pd.concat(
                [df_regular_continuidad_01,
                 df_regular_continuidad_estacional_03],
                ignore_index=True)

        else:
            df_regular_continuidad = df_regular_full[
                es_base & ~es_estacional & ~es_fin
                ].copy()

            df_regular_continuidad_01 = df_regular_continuidad[
                ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'POSTERIOR_(+1)']].drop_duplicates()

            df_regular_continuidad_full = df_regular_continuidad_01.copy()

        df_regular_continuidad_full['PERIODO_TARGET'] = periodo
        df_regular_continuidad_full['PERIODO_LAG_12'] = lag_12

        df_regular_continuidad_full['IDX'] = 1
        df_regular_continuidad_full_01 = self.add_pe(df_regular_continuidad_full, 
                                                     periodo_column='PERIODO')
        
        df_regular_continuidad_full_02 = df_regular_continuidad_full_01.rename(
            columns={"PE":'PE_ANTERIOR'}
        )

        df_regular_continuidad_full_03 = df_regular_continuidad_full_02.rename(
            columns={'CURSO_ACTUAL':'CURSO_ANTERIOR'}
        )

        df_regular_continuidad_full_04 = df_regular_continuidad_full_03.rename(
            columns={'POSTERIOR_(+1)': 'CURSO_ACTUAL'}
        )

        df_regular_continuidad_full_05 = self.add_pe(df_regular_continuidad_full_04, 
                                                     periodo_column='PERIODO_TARGET')
        df_regular_continuidad_full_06 = self.add_vac_estandar(df_regular_continuidad_full_05,
                                                               periodo_column='PERIODO_TARGET')

        df_regular_continuidad_full_07 = self.add_flags(df_regular_continuidad_full_06)
        
        df_regular_continuidad_full_08 = df_regular_continuidad_full_07[
            df_regular_continuidad_full_07['FLAG_INICIAL'] == 0].copy()


        df_regular_continuidad_full_09 = df_regular_continuidad_full_08[
            ['PERIODO_TARGET', 'PERIODO', 'PERIODO_LAG_12', 'SEDE', 'CURSO_ANTERIOR',
             'CURSO_ACTUAL', 'IDX', 'PE', 'VAC_ACAD_ESTANDAR', 'PE_ANTERIOR']].copy()

        return df_regular_continuidad_full_09

    def get_idx_predict(self, periodo):
        """
        Obtiene el índice de continuidad regular para un periodo dado.
        """
        idx = self.get_idx(periodo)
        idx_01 = idx.drop_duplicates(
            ['PERIODO_TARGET', 'PERIODO', 'SEDE',
             'CURSO_ACTUAL', 'IDX']).copy()
        return idx_01

    def add_anterior(self, df_idx: pd.DataFrame):
        """
        Agrega información del periodo anterior.
        """
        df_regular_full = self.df_regular_full[
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL',
             'CANT_CLASES', 'CANT_ALUMNOS']].copy()

        df_regular_full_01 = df_regular_full.groupby(
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL'], as_index=False
        )[['CANT_CLASES', 'CANT_ALUMNOS']].sum()

        df_regular_full_02 = df_regular_full_01.rename(
            columns={'CURSO_ACTUAL':'CURSO_ANTERIOR'})

        df_idx_01 = df_idx.merge(
            df_regular_full_02,
            on=['PERIODO', 'SEDE', 'CURSO_ANTERIOR'],
            how='left')

        df_idx_01['CANT_CLASES'] = df_idx_01['CANT_CLASES'].fillna(0)
        df_idx_01['CANT_ALUMNOS'] = df_idx_01['CANT_ALUMNOS'].fillna(0)

        df_idx_02 = df_idx_01.groupby(
            ['PERIODO_TARGET', 'PERIODO_LAG_12', 'SEDE', 'CURSO_ACTUAL','IDX', 'PE', 'PE_ANTERIOR',
            'VAC_ACAD_ESTANDAR'], as_index=False
        ).agg(
            CANT_CLASES_ANTERIOR=pd.NamedAgg(
                column='CANT_CLASES',
                aggfunc='sum'
            ),
            CANT_ALUMNOS_ANTERIOR=pd.NamedAgg(
                column='CANT_ALUMNOS',
                aggfunc='sum'
            )
        )

        return df_idx_02

    def add_lag_n(self, df_idx: pd.DataFrame, n:int):
        """
        Agrega información del periodo lag n.
        """
        df_regular_full = self.df_regular_full[
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL',
             'CANT_CLASES', 'CANT_ALUMNOS']].copy()

        df_regular_full_01 = df_regular_full.groupby(
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL'], as_index=False
        )[['CANT_CLASES', 'CANT_ALUMNOS']].sum()

        assert 12 >= n >= 1
        lag_n = str(n).zfill(2)

        lag_n_df_regular = df_regular_full_01.rename(
            columns={'PERIODO':f'PERIODO_LAG_{lag_n}',
                     'CANT_CLASES': f'CANT_CLASES_LAG_{lag_n}',
                     'CANT_ALUMNOS': f'CANT_ALUMNOS_LAG_{lag_n}',
                     }
        )

        df_idx_01 = df_idx.merge(
            lag_n_df_regular,
            on=[f'PERIODO_LAG_{lag_n}', 'SEDE', 'CURSO_ACTUAL'],
            how='left'
        )
        df_idx_01[f'CANT_CLASES_LAG_{lag_n}'] = df_idx_01[f'CANT_CLASES_LAG_{lag_n}'].fillna(0)
        df_idx_01[f'CANT_ALUMNOS_LAG_{lag_n}'] = df_idx_01[f'CANT_ALUMNOS_LAG_{lag_n}'].fillna(0)

        return df_idx_01

    def add_diff_sply(self, df_idx: pd.DataFrame):
        """
        Agrega información de diferencia de oferta entre el periodo target y el periodo lag 12.
        """
        periodo = df_idx.PERIODO_TARGET.unique()[0]
        lag_12 = get_n_lags(periodo, 12)
        lag_12_df_idx = self.get_idx(lag_12)
        lag_12_df_idx_01 = self.add_anterior(lag_12_df_idx)
        columns = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 
                   'CANT_CLASES_ANTERIOR', 'CANT_ALUMNOS_ANTERIOR']
        lag_12_df_idx_02 = lag_12_df_idx_01[columns].copy()
        
        target_lag_12 = self.get_target(lag_12)
        columns_target = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'CANT_CLASES', 'CANT_ALUMNOS']
        target_lag_12 = target_lag_12[columns_target].copy()
        
        lag_12_df_idx_03 = lag_12_df_idx_02.merge(
            target_lag_12,
            on=['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL'],
            how='left'
        )
        lag_12_df_idx_03['CANT_ALUMNOS'] = lag_12_df_idx_03['CANT_ALUMNOS'].fillna(0)
        lag_12_df_idx_03['CANT_CLASES'] = lag_12_df_idx_03['CANT_CLASES'].fillna(0)
        lag_12_df_idx_03['DIFF_ALUMNOS_SPLY'] = lag_12_df_idx_03['CANT_ALUMNOS'] - lag_12_df_idx_03['CANT_ALUMNOS_ANTERIOR']
        lag_12_df_idx_03['PCT_ALUMNOS_SPLY'] = lag_12_df_idx_03['CANT_ALUMNOS']/lag_12_df_idx_03['CANT_ALUMNOS_ANTERIOR'] - 1
        lag_12_df_idx_03['DIFF_CLASES_SPLY'] = lag_12_df_idx_03['CANT_CLASES'] - lag_12_df_idx_03['CANT_CLASES_ANTERIOR']
        lag_12_df_idx_03['PCT_CLASES_SPLY'] = lag_12_df_idx_03['CANT_CLASES']/lag_12_df_idx_03['CANT_CLASES_ANTERIOR'] - 1
        lag_12_df_idx_04 = lag_12_df_idx_03[
            ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 
            'DIFF_ALUMNOS_SPLY', 'PCT_ALUMNOS_SPLY',
            'DIFF_CLASES_SPLY', 'PCT_CLASES_SPLY']].copy()

        lag_12_df_idx_04['PERIODO_TARGET'] = periodo

        df_idx_01 = df_idx.merge(
            lag_12_df_idx_04,
            on=['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL'],
            how='left'
        )

        df_idx_01['DIFF_ALUMNOS_SPLY'] = df_idx_01['DIFF_ALUMNOS_SPLY'].fillna(0)
        df_idx_01['PCT_ALUMNOS_SPLY'] = df_idx_01['PCT_ALUMNOS_SPLY'].fillna(0)
        df_idx_01['DIFF_CLASES_SPLY'] = df_idx_01['DIFF_CLASES_SPLY'].fillna(0)
        df_idx_01['PCT_CLASES_SPLY'] = df_idx_01['PCT_CLASES_SPLY'].fillna(0)

        return df_idx_01
    
    def add_quantitative_feats(self, df_idx):
        """
        Agrega features cuantitativas al índice.
        """
        df_idx_01 = self.add_anterior(df_idx)
        df_idx_02 = self.add_lag_n(df_idx_01, 12)
        df_idx_03 = self.add_diff_sply(df_idx_02)
        return df_idx_03

    def add_categorical_feats(self, df_idx):
        """
        Agrega features categóricas al índice.
        """
        df_idx_01 = df_idx.merge(
            self.df_curso_actual[['CURSO_ACTUAL', 'NIVEL', 'CURSO_2', 'FRECUENCIA',
                                  'DURACION']].copy(),
            on=['CURSO_ACTUAL'],
            how='left'
        )

        assert df_idx_01['CURSO_ACTUAL'].isnull().sum() == 0

        return df_idx_01

    def get_target(self, periodo: int):
        """
        Obtiene el target de continuidad regular para un periodo dado.
        """
        df_real = self.df_real.copy()
        df_real_01 = df_real[
            (df_real['PERIODO'] == periodo)
            & (df_real['FLAG_INICIAL'] == 0)].copy()

        df_real_02 = df_real_01.groupby(
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL'], as_index=False
        )[['CANT_CLASES', 'CANT_ALUMNOS']].sum()

        df_real_03 = df_real_02.rename(
                columns={'PERIODO':'PERIODO_TARGET'}
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
        """
        Obtiene el modelo de continuidad regular para un periodo dado.
        """
        data_idx = self.get_idx(periodo)
        data_target = self.get_target(periodo)

        data_quantitative = self.add_quantitative_feats(data_idx)
        data_model = data_quantitative.merge(
            data_target,
            on=['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL'],
            how='left'
        )

        data_model['CANT_CLASES'] = data_model['CANT_CLASES'].fillna(0).astype('int32')
        data_model['CANT_ALUMNOS'] = data_model['CANT_ALUMNOS'].fillna(0).astype('int32')

        data_model['PCT_ALUMNOS'] = data_model['CANT_ALUMNOS']/data_model['CANT_ALUMNOS_ANTERIOR'] - 1
        data_model['PCT_CLASES'] = data_model['CANT_CLASES']/data_model['CANT_CLASES_ANTERIOR'] - 1

        return data_model

    def get_model(self, periodos: list[int]):
        """
        Obtiene el modelo de continuidad regular para un periodo dado.
        """
        collection = []
        for periodo in periodos:
            # print(periodo)
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
        """
        Obtiene las features de continuidad regular para un periodo dado.
        """
        periodos = get_training_periodos(periodo)
        data_model = self.get_model(periodos)

        columns_categorical = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']
        
        data_model_01 = data_model.copy()
        
        # data_model_01['FLAG_PE'] = np.where(
        #     data_model_01['CANT_ALUMNOS_ANTERIOR'] == data_model_01['PE_ANTERIOR'], 1, 0)

        data_model_01 = data_model_01[
            ~(
                ((data_model_01['SEDE'] == SEDES[0]) & 
                (data_model_01['PERIODO_TARGET'].isin(SKIPPED_FEATURES_PERIODOS[0])))|
                ((data_model_01['SEDE'] == SEDES[1]) & 
                (data_model_01['PERIODO_TARGET'].isin(SKIPPED_FEATURES_PERIODOS[1])))
            )
            ].copy()
        
        # self.dummy_columns = dummy_columns.copy() + ['FLAG_PE']
        
        data_model_train = data_model_01[data_model_01.PERIODO_TARGET < periodo].copy()
        data_model_test = data_model_01[data_model_01.PERIODO_TARGET  == periodo].copy()

        return (data_model_train, data_model_test)

    def load_features(self, periodo:int, version:str, output_feats_continuidad_datastore:str,with_tipo:str):
        """
        Carga las features de continuidad regular para un periodo dado.
        """
        eval_tipo = eval(with_tipo)
        
        if not eval_tipo:
            output_feats_continuidad_datastore = f"{output_feats_continuidad_datastore}/{self.tipo}"
        
        data_model_train, data_model_test = self.get_features(periodo)
        data_model_train.to_parquet(
            f"{output_feats_continuidad_datastore}/train/{version}/data_feats_{self.tipo}_{periodo}.parquet", index=False)
        data_model_test.to_parquet(
            f"{output_feats_continuidad_datastore}/test/{version}/data_feats_{self.tipo}_{periodo}.parquet", index=False)

    def load_target(self, periodo:int, version:str, output_target_continuidad_datastore:str,with_tipo:str):
        """
        Carga el target de continuidad regular para un periodo dado.
        """
        eval_tipo = eval(with_tipo)
        
        if not eval_tipo:
            output_target_continuidad_datastore = f"{output_target_continuidad_datastore}/{self.tipo}"

        try:
            df_real_09 = self.get_target(periodo)
            df_real_09.to_parquet(
                f"{output_target_continuidad_datastore}/test/{version}/data_target_{self.tipo}_{periodo}.parquet", index=False)
        except Exception as e:
            print(f"Error al cargar target para periodo {periodo}")
            print(e)


def main(args):
    """
    Función principal para ejecutar el pipeline de continuidad regular.
    """
    input_datastore=args.input_datastore
    ult_periodo=args.ult_periodo

    mode = args.mode
    periodo = args.periodo
    with_tipo = args.with_tipo
    
    # for Continuidad
    output_feats_continuidad_datastore = args.output_feats_datastore
    output_target_continuidad_datastore = args.output_target_datastore
    platinum_version = args.platinum_version
    feats_version=args.feats_version
    target_version=args.target_version
    n_eval_periodos=args.n_eval_periodos
    model_periodo=args.model_periodo
    
    
    loader = Loader(
        input_datastore, 
        platinum_version,
        ult_periodo)
    
    tablas = loader.fetch_all()

    # Continuidad a nivel de curso
    continuidad = Continuidad(tablas)
    tipo_continuidad = continuidad.tipo

    feats_train_path = Path(output_feats_continuidad_datastore)  / "train" / feats_version
    feats_test_path = Path(output_feats_continuidad_datastore)  / "test" / feats_version
    target_test_path = Path(output_target_continuidad_datastore)  / "test" / target_version
    
    feats_train_path.mkdir(parents=True, exist_ok=True)
    feats_test_path.mkdir(parents=True, exist_ok=True)
    target_test_path.mkdir(parents=True, exist_ok=True)

    if (periodo != -1) and (n_eval_periodos == -1):
        periodos = [periodo]
    else:
        assert n_eval_periodos >= 1, "n_eval_periodos debe ser mayor o igual a 1"
        periodos = get_ahead_n_periodos(model_periodo, n_eval_periodos)
    
    
    for periodo in periodos:
        mapping_tipos = get_mapping_tipos(periodo)

        # Create directories for Continuidad
        
        if mapping_tipos[tipo_continuidad]:

            continuidad.load_features(
                periodo, 
                feats_version,
                output_feats_continuidad_datastore, with_tipo)
        
            continuidad.load_target(
                periodo, 
                target_version,
                output_target_continuidad_datastore, with_tipo)
        else:
            print(f"No se generaron features para el periodo {periodo} y tipo {tipo_continuidad}")


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
    