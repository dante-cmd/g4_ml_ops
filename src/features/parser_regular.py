
"""
Parser Regular - Parseo de datos regulares
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
from pathlib import Path
from unidecode import unidecode


class Utils:
    def __init__(self, tablas:dict):
        self.df_prog_acad = tablas['prog_acad'].copy()
        self.df_synthetic = tablas['synthetic'].copy()
        self.df_curso_actual = tablas['tabla_curso_actual'].copy()
        self.df_curso_acumulado = tablas['tabla_curso_acumulado'].copy()
        self.df_curso_inicial = tablas['tabla_curso_inicial'].copy()
        self.df_vac_estandar = tablas['tabla_vac_estandar'].copy()
        self.df_pe = tablas['tabla_pe'].copy()
        self.df_horario_diario_to_sabatino = tablas['tabla_horario_diario_to_sabatino'].copy()
        self.df_curso_diario_to_sabatino = tablas['tabla_curso_diario_to_sabatino'].copy()
        self.df_turno_disponible = tablas['tabla_turno_disponible'].copy()
        self.df_horario = tablas['tabla_horario'].copy()

    def add_flags(self, df:pd.DataFrame):
        """
        Docstring for add_flags
        Add flags like 'FLAG_INICIAL', 'FLAG_ESTACIONAL' and the column 'CURSO_2'
        
        :param self: Description
        :param df: Description
        :type df: pd.DataFrame
        """
        assert np.isin(['CURSO_ACTUAL'], df.columns).all()
        assert not np.isin(
            ['FLAG_ESTACIONAL', 'FLAG_INICIAL', 'CURSO_2'], df.columns).any()
        df_01 = df.merge(
            self.df_curso_acumulado[
                ['CURSO_ANTERIOR', 'FLAG_ESTACIONAL']].rename(
                columns={'CURSO_ANTERIOR': 'CURSO_ACTUAL'}
            ),
            on='CURSO_ACTUAL',
            how='left'
        )
        
        assert df_01['FLAG_ESTACIONAL'].isnull().sum() == 0

        df_02 = df_01.merge(
            self.df_curso_actual[
                ['CURSO_ACTUAL', 'CURSO_2']].copy(),
            on=['CURSO_ACTUAL'],
            how='left')
        
        assert df_02['CURSO_2'].isnull().sum() == 0

        df_02['FLAG_INICIAL'] = np.where(
            df_02['CURSO_2'].isin(self.df_curso_inicial['CURSOS_INICIALES']), 1, 0)
        
        df_03 = df_02.drop(columns=['CURSO_2'])

        return df_03
    
    def add_pe(self, df:pd.DataFrame, periodo_column:str= 'PERIODO'):

        assert np.isin(['CURSO_ACTUAL', periodo_column], df.columns).all()
        assert not np.isin(['PROGRAMA', 'PE'], df.columns).any()

        df_01 = df.merge(
            self.df_curso_actual[
                ['CURSO_ACTUAL', 'PROGRAMA']].copy(),
            on=['CURSO_ACTUAL'],
            how='left'
        )

        assert df_01['PROGRAMA'].isnull().sum() == 0

        df_01['PROGRAMA'] = np.where(
            df_01['PROGRAMA'] == 'Niños',
            'Niños',
            'Adultos'
        )

        df_pe = self.df_pe.rename(
            columns={"PERIODO":periodo_column}
        )

        df_pe = df_pe[
            [periodo_column, 'PROGRAMA', 'PE']].copy()
        
        df_02 = df_01.merge(
            df_pe,
            on=[periodo_column, 'PROGRAMA'],
            how='left'
        )

        assert df_02['PE'].isnull().sum() == 0
        
        df_03 = df_02.drop(columns=['PROGRAMA'])

        return df_03
    
    def add_vac_estandar(self, df:pd.DataFrame, periodo_column:str= 'PERIODO'):
        assert np.isin(['CURSO_ACTUAL', periodo_column], df.columns).all()
        assert not np.isin(['NIVEL', 'LINEA_DE_NEGOCIO', 'VAC_ACAD_ESTANDAR'], df.columns).any()

        df_01 = df.merge(
            self.df_curso_actual[
                ['CURSO_ACTUAL', 'NIVEL', 'LINEA_DE_NEGOCIO']].copy(),
            on=['CURSO_ACTUAL'],
            how='left'
        )

        assert df_01['NIVEL'].isnull().sum() == 0

        df_vac_estandar = self.df_vac_estandar.rename(
            columns={"PERIODO":periodo_column}
        )
        df_vac_estandar = df_vac_estandar[
            [periodo_column, 'NIVEL', 'LINEA_DE_NEGOCIO', 'VAC_ACAD_ESTANDAR']].copy()

        df_02 = df_01.merge(
            df_vac_estandar,
            on=[periodo_column, 'NIVEL', 'LINEA_DE_NEGOCIO'],
            how='left'
        )

        assert df_02['VAC_ACAD_ESTANDAR'].isnull().sum() == 0

        df_03 = df_02.drop(columns=['NIVEL', 'LINEA_DE_NEGOCIO'])

        return df_03
    
    def get_regular(self):
        # synth
        df_prog_acad = self.df_prog_acad.copy()

        df_prog_acad_01 = df_prog_acad.groupby(
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL'],
            as_index=False, dropna=False
        ).agg(
            CANT_ALUMNOS=pd.NamedAgg(
                column='CANT_ALUMNOS',
                aggfunc='sum'
            ),
            CANT_CLASES=pd.NamedAgg(
                column='CANT_CLASES',
                aggfunc='sum'
            )
        )

        df_prog_acad_02 = self.add_flags(df_prog_acad_01)

        df_prog_acad_03 = df_prog_acad_02[df_prog_acad_02['FLAG_ESTACIONAL'] == 0].copy()

        return df_prog_acad_03
    
    def get_regular_plus_synth(self):
        # synth
        if self.df_synthetic.empty:
            df_prog_acad = self.df_prog_acad.copy()
        else:
            df_prog_acad = pd.concat(
                [self.df_prog_acad, self.df_synthetic],
                ignore_index=True
            )

        df_prog_acad_01 = df_prog_acad.groupby(
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL'],
            as_index=False, dropna=False
        ).agg(
            CANT_ALUMNOS=pd.NamedAgg(
                column='CANT_ALUMNOS',
                aggfunc='sum'
            ),
            CANT_CLASES=pd.NamedAgg(
                column='CANT_CLASES',
                aggfunc='sum'
            )
        )

        df_prog_acad_02 = self.add_flags(df_prog_acad_01)

        df_prog_acad_03 = df_prog_acad_02[
            df_prog_acad_02['FLAG_ESTACIONAL'] == 0
            ].copy()

        return df_prog_acad_03

    def get_regular_plus_synth_plus_estacional(self):
        # synth
        if self.df_synthetic.empty:
            df_prog_acad = self.df_prog_acad.copy()
        else:
            df_prog_acad = pd.concat([self.df_prog_acad, self.df_synthetic], ignore_index=True)

        df_prog_acad_01 = df_prog_acad.groupby(
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL'],
            as_index=False, dropna=False
        ).agg(
            CANT_ALUMNOS=pd.NamedAgg(
                column='CANT_ALUMNOS',
                aggfunc='sum'
            ),
            CANT_CLASES=pd.NamedAgg(
                column='CANT_CLASES',
                aggfunc='sum'
            )
        )

        df_prog_acad_02 = self.add_flags(df_prog_acad_01)

        return df_prog_acad_02

    