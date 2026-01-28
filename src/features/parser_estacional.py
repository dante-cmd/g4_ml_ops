
"""
Parser Estacional - Parseo de datos estacionales
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
from pathlib import Path


class Utils:
    def __init__(self, tablas:dict):
        self.df_prog_acad = tablas['prog_acad'].copy()
        self.df_synthetic = tablas['synthetic'].copy()
        self.df_curso_actual = tablas['tabla_curso_actual'].copy()
        self.df_curso_acumulado = tablas['tabla_curso_acumulado'].copy()
        self.df_curso_inicial = tablas['tabla_curso_inicial'].copy()
        self.df_vac_estandar = tablas['tabla_vac_estandar'].copy()
        self.df_pe = tablas['tabla_pe'].copy()
        self.df_turno_disponible = tablas['tabla_turno_disponible'].copy()
        self.df_horario = tablas['tabla_horario'].copy()

    def get_estacional_plus_synth(self):
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

        df_prog_acad_02 = df_prog_acad_01.merge(
            self.df_curso_acumulado[
                ['CURSO_ANTERIOR', 'FLAG_ESTACIONAL']].rename(
                columns={'CURSO_ANTERIOR': 'CURSO_ACTUAL'}
            ),
            on='CURSO_ACTUAL',
            how='left'
        )

        assert df_prog_acad_02['FLAG_ESTACIONAL'].isnull().sum() == 0

        df_prog_acad_03 = df_prog_acad_02[df_prog_acad_02['FLAG_ESTACIONAL'] == 1].copy()

        df_prog_acad_04 = df_prog_acad_03.merge(
            self.df_curso_actual[
                ['CURSO_ACTUAL', 'PROGRAMA', 'NIVEL', 'LINEA_DE_NEGOCIO', 'CURSO_2',
                 'POSTERIOR_(+1)']
            ].copy(),
            on=['CURSO_ACTUAL'],
            how='left'
        )
        
        assert df_prog_acad_04['PROGRAMA'].isnull().sum() == 0

        df_prog_acad_04['PROGRAMA'] = np.where(
            df_prog_acad_04['PROGRAMA'] == 'Niños',
            'Niños',
            'Adultos'
        )
        
        df_prog_acad_04['FLAG_INICIAL'] = np.where(
            df_prog_acad_04['CURSO_2'].isin(
                self.df_curso_inicial['CURSOS_INICIALES']), 1, 0)
        
        df_vac_estandar = self.df_vac_estandar.copy()

        df_prog_acad_05 = df_prog_acad_04.merge(
            df_vac_estandar,
            on=['PERIODO', 'LINEA_DE_NEGOCIO', 'NIVEL'],
            how='left'
        )
        assert df_prog_acad_05['VAC_ACAD_ESTANDAR'].isnull().sum() == 0

        df_pe = self.df_pe.copy()

        df_prog_acad_06 = df_prog_acad_05.merge(
            df_pe,
            on=['PERIODO', 'PROGRAMA'],
            how='left'
        )

        assert df_prog_acad_06['PE'].isnull().sum() == 0

        df_prog_acad_07 = df_prog_acad_06[
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL', 'FLAG_INICIAL',
            'PE', 'VAC_ACAD_ESTANDAR', 'CANT_ALUMNOS', 'CANT_CLASES', 'POSTERIOR_(+1)']].copy()

        return df_prog_acad_07

    def get_estacional(self):
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

        df_prog_acad_02 = df_prog_acad_01.merge(
            self.df_curso_acumulado[
                ['CURSO_ANTERIOR', 'FLAG_ESTACIONAL', 'CURSO_2']].rename(
                columns={'CURSO_ANTERIOR': 'CURSO_ACTUAL'}
            ),
            on='CURSO_ACTUAL',
            how='left'
        )

        df_prog_acad_02['FLAG_INICIAL'] = np.where(
            df_prog_acad_02['CURSO_2'].isin(self.df_curso_inicial['CURSOS_INICIALES']), 1, 0)

        assert df_prog_acad_02['FLAG_ESTACIONAL'].isnull().sum() == 0

        df_prog_acad_03 = df_prog_acad_02[df_prog_acad_02['FLAG_ESTACIONAL'] == 1].copy()

        return df_prog_acad_03

