import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from unidecode import unidecode
import os
import argparse
from pathlib import Path
from utils_data import validate_periodos


class Etl:

    def __init__(self, 
                 input_datastore: str, 
                 output_datastore: str,
                 platinum_version: str,
                 ult_periodo: int, 
                 n_periodos: int | None = 3,
                 all_periodos: bool = False) -> None:
        
        self.input_datastore = Path(input_datastore)
        self.output_datastore = Path(output_datastore)
        self.platinum_version = platinum_version
        self.periodo = ult_periodo
        self.n_periodos = n_periodos
        self.all_periodos = all_periodos
        self.periodos = validate_periodos(n_periodos, ult_periodo, all_periodos)

    def fetch_tabla_horario(self):
        path = self.input_datastore / 'dim'/'Horarios Merge.xlsx'
        tabla_horario = pd.read_excel(path, sheet_name='Hoja1')

        tabla_horario.columns = tabla_horario.columns.astype('string')
        tabla_horario.columns = tabla_horario.columns.str.strip().str.upper()
        tabla_horario.columns = tabla_horario.columns.str.replace('.', '')
        tabla_horario.columns = tabla_horario.columns.str.replace(' ', '_').map(unidecode)
        tabla_horario_normal = tabla_horario.melt(
            id_vars=['HORARIO'],
            value_vars=['TURNO_1', 'TURNO_2', 'TURNO_3', 'TURNO_4'],
            var_name='N_TURNO',
            value_name='FRANJA'
        )
        tabla_horario_normal_01 = tabla_horario_normal[~tabla_horario_normal.FRANJA.isnull()].copy()
        hora_inicial = tabla_horario_normal_01['FRANJA'].str.replace(
            r"(\d{2})\:(\d{2}) - (\d{2})\:(\d{2})", r"\1", regex=True).astype('int32')
        minuto_inicial = tabla_horario_normal_01['FRANJA'].str.replace(
            r"(\d{2})\:(\d{2}) - (\d{2})\:(\d{2})", r"\2", regex=True).astype('int32')
        hora_final = tabla_horario_normal_01['FRANJA'].str.replace(
            r"(\d{2})\:(\d{2}) - (\d{2})\:(\d{2})", r"\3", regex=True).astype('int32')
        minuto_final = tabla_horario_normal_01['FRANJA'].str.replace(
            r"(\d{2})\:(\d{2}) - (\d{2})\:(\d{2})", r"\4", regex=True).astype('int32')
        tabla_horario_normal_01['HORA_INICIO'] = hora_inicial + (minuto_inicial/60)
        tabla_horario_normal_01['HORA_FIN'] = hora_final + (minuto_final/60)

        tabla_horario_normal_02 = tabla_horario_normal_01.groupby(
            ['HORARIO'], as_index=False
        ).agg(
            HORA_MIN=pd.NamedAgg(
                column='HORA_INICIO',
                aggfunc='min'
                ),
        HORA_MAX=pd.NamedAgg(
            column='HORA_FIN',
            aggfunc='max'
            )
    )

        tabla_horario_01 = tabla_horario.merge(
            tabla_horario_normal_02,
            on='HORARIO',
            how='left'
        )

        return tabla_horario_01

    def fetch_tabla_curso_actual(self):
        path = self.input_datastore / 'dim'/'Cursos Actual.xlsx'
        tabla_curso_actual = pd.read_excel(path, sheet_name='BBDD')
        
        tabla_curso_actual.columns = tabla_curso_actual.columns.astype('string')
        tabla_curso_actual.columns = tabla_curso_actual.columns.str.strip().str.upper()
        tabla_curso_actual.columns = tabla_curso_actual.columns.str.replace('.', '')
        tabla_curso_actual.columns = tabla_curso_actual.columns.str.replace(' ', '_').map(unidecode)

        return tabla_curso_actual

    def fetch_tabla_curso_acumulado(self):
        path = self.input_datastore / 'dim'/'Cursos Acumulado.xlsx'
        
        tabla_curso_acumulado = pd.read_excel(path, sheet_name='BBDD')
        
        tabla_curso_acumulado.columns = tabla_curso_acumulado.columns.astype('string')
        tabla_curso_acumulado.columns = tabla_curso_acumulado.columns.str.strip().str.upper()
        tabla_curso_acumulado.columns = tabla_curso_acumulado.columns.str.replace('.', '')
        tabla_curso_acumulado.columns = tabla_curso_acumulado.columns.str.replace(' ', '_').map(unidecode)
        return tabla_curso_acumulado

    def fetch_tabla_curso_estacional(self):
        path = self.input_datastore / 'dim'/'Cursos Actual.xlsx'
        
        tabla_curso_estacional = pd.read_excel(path, sheet_name='ESTACIONAL')
        
        tabla_curso_estacional.columns = tabla_curso_estacional.columns.astype('string')
        tabla_curso_estacional.columns = tabla_curso_estacional.columns.str.strip().str.upper()
        tabla_curso_estacional.columns = tabla_curso_estacional.columns.str.replace('.', '')
        tabla_curso_estacional.columns = tabla_curso_estacional.columns.str.replace(' ', '_').map(unidecode)
        return tabla_curso_estacional

    def fetch_tabla_sede(self):

        path = self.input_datastore / 'dim'/'Total Aulas.xlsx'
        
        tabla_sede = pd.read_excel(path, sheet_name='SEDE')
        
        tabla_sede.columns = tabla_sede.columns.astype('string')
        tabla_sede.columns = tabla_sede.columns.str.strip().str.upper()
        tabla_sede.columns = tabla_sede.columns.str.replace('.', '')
        tabla_sede.columns = tabla_sede.columns.str.replace(' ', '_').map(unidecode)
        return tabla_sede

    def fetch_tabla_curso_inicial(self):
        path = self.input_datastore / 'dim'/'Cursos Iniciales.xlsx'
        
        tabla_curso_inicial = pd.read_excel(path, sheet_name='CURSOS_INICIALES')
        
        tabla_curso_inicial.columns = tabla_curso_inicial.columns.astype('string')
        tabla_curso_inicial.columns = tabla_curso_inicial.columns.str.strip().str.upper()
        tabla_curso_inicial.columns = tabla_curso_inicial.columns.str.replace('.', '')
        tabla_curso_inicial.columns = tabla_curso_inicial.columns.str.replace(' ', '_').map(unidecode)
        return tabla_curso_inicial

    def fetch_tabla_aula(self):
        path = self.input_datastore / 'dim'/ 'Total Aulas.xlsx'
        
        tabla_aula = pd.read_excel(path, sheet_name='BBDD')
        
        tabla_aula.columns = tabla_aula.columns.astype('string')
        tabla_aula.columns = tabla_aula.columns.str.strip().str.upper()
        tabla_aula.columns = tabla_aula.columns.str.replace('.', '')
        tabla_aula.columns = tabla_aula.columns.str.replace(' ', '_').map(unidecode)
        return tabla_aula

    def fetch_tabla_turno_disponible(self):
        path = self.input_datastore / 'dim'/ 'Total Aulas.xlsx'
        
        tabla_turno_disponible = pd.read_excel(path, sheet_name='HORARIOS_ATENCION')
        
        tabla_turno_disponible.columns = tabla_turno_disponible.columns.astype('string')
        tabla_turno_disponible.columns = tabla_turno_disponible.columns.str.strip().str.upper()
        tabla_turno_disponible.columns = tabla_turno_disponible.columns.str.replace('.', '')
        tabla_turno_disponible.columns = tabla_turno_disponible.columns.str.replace(' ', '_').map(unidecode)

        columns = list(tabla_turno_disponible.columns[tabla_turno_disponible.columns.str.contains(r'\d')])
        tabla_turno_disponible_normal = tabla_turno_disponible.melt(
            id_vars=['SEDE', 'PERIODO_FRANJA', 'FRANJA', 'YEAR'],
            value_vars=columns,
            var_name='MONTH',
            value_name='FLAG_DISPONIBLE'
        )
        tabla_turno_disponible_normal_01 = tabla_turno_disponible_normal[
            tabla_turno_disponible_normal.FLAG_DISPONIBLE != 0].copy()
        tabla_turno_disponible_normal_01['PERIODO'] = (
            tabla_turno_disponible_normal_01['YEAR'].astype('string') + 
            tabla_turno_disponible_normal_01['MONTH'].astype('string').str.zfill(2))
        tabla_turno_disponible_normal_01['PERIODO'] = tabla_turno_disponible_normal_01['PERIODO'].astype('int32')
        tabla_turno_disponible_normal_02 = tabla_turno_disponible_normal_01[
            ['PERIODO', 'SEDE', 'PERIODO_FRANJA', 'FRANJA', 'FLAG_DISPONIBLE']].copy()
        hora_inicial = tabla_turno_disponible_normal_02['FRANJA'].str.replace(
            r"(\d{2})\:(\d{2}) - (\d{2})\:(\d{2})", r"\1", regex=True).astype('int32')
        minuto_inicial = tabla_turno_disponible_normal_02['FRANJA'].str.replace(
            r"(\d{2})\:(\d{2}) - (\d{2})\:(\d{2})", r"\2", regex=True).astype('int32')
        hora_final = tabla_turno_disponible_normal_02['FRANJA'].str.replace(
            r"(\d{2})\:(\d{2}) - (\d{2})\:(\d{2})", r"\3", regex=True).astype('int32')
        minuto_final = tabla_turno_disponible_normal_02['FRANJA'].str.replace(
            r"(\d{2})\:(\d{2}) - (\d{2})\:(\d{2})", r"\4", regex=True).astype('int32')
        tabla_turno_disponible_normal_02['HORA_INICIAL'] = hora_inicial + (minuto_inicial/60)
        tabla_turno_disponible_normal_02['HORA_FINAL'] = hora_final + (minuto_final/60)
        horas_de_atencion = tabla_turno_disponible_normal_02.groupby(
            ['PERIODO', 'SEDE'], as_index=False
            ).agg(
                HORA_MIN=pd.NamedAgg(
                    column='HORA_INICIAL',
                    aggfunc='min'
                    ),
                HORA_MAX=pd.NamedAgg(
                    column='HORA_FINAL',
                    aggfunc='max'
                    )
            )
        return horas_de_atencion

    def fetch_tabla_vac_estandar(self):
        path = self.input_datastore / 'dim'/'Total Aulas.xlsx'
        
        tabla_vac_estandar = pd.read_excel(path, sheet_name='VAC_ACAD_ESTANDAR')
        
        tabla_vac_estandar.columns = tabla_vac_estandar.columns.astype('string')
        tabla_vac_estandar.columns = tabla_vac_estandar.columns.str.strip().str.upper()
        tabla_vac_estandar.columns = tabla_vac_estandar.columns.str.replace('.', '')
        tabla_vac_estandar.columns = tabla_vac_estandar.columns.str.replace(' ', '_').map(unidecode)
        return tabla_vac_estandar

    def fetch_tabla_pe(self):
        path = self.input_datastore / 'dim'/'Total Aulas.xlsx'
        tabla_pe = pd.read_excel(path, sheet_name='PE')
     
        tabla_pe.columns = tabla_pe.columns.astype('string')
        tabla_pe.columns = tabla_pe.columns.str.strip().str.upper()
        tabla_pe.columns = tabla_pe.columns.str.replace('.', '')
        tabla_pe.columns = tabla_pe.columns.str.replace(' ', '_').map(unidecode)
        return tabla_pe

    def fetch_synthetic(self):
        path = self.input_datastore / 'synthetic'/'synthetic.xlsx'
        synthetic = pd.read_excel(path)
        synthetic.columns = synthetic.columns.astype('string')
        synthetic.columns = synthetic.columns.str.strip().str.upper()
        synthetic.columns = synthetic.columns.str.replace('.', '')
        synthetic.columns = synthetic.columns.str.replace(' ', '_').map(unidecode)
        return synthetic

    def fetch_tabla_curso_diario_to_sabatino(self):

        path = self.input_datastore / 'dim'/'Cursos Actual.xlsx'
        tabla_estacional = pd.read_excel(path, sheet_name='ESTACIONAL')
        tabla_estacional.columns = tabla_estacional.columns.astype('string')
        tabla_estacional.columns = tabla_estacional.columns.str.strip().str.upper()
        tabla_estacional.columns = tabla_estacional.columns.str.replace('.', '')
        tabla_estacional.columns = tabla_estacional.columns.str.replace(' ', '_').map(unidecode)
        tabla_estacional_01 = tabla_estacional[['CURSO_ACTUAL', 'POSTERIOR_(+1)']].copy()
        return tabla_estacional_01

    def fetch_tabla_horario_diario_to_sabatino(self):
        path = self.input_datastore / 'dim'/'Horarios Merge.xlsx'
        tabla_horario_estacional = pd.read_excel(path, sheet_name='ESTACIONAL')

        tabla_horario_estacional.columns = tabla_horario_estacional.columns.astype('string')
        tabla_horario_estacional.columns = tabla_horario_estacional.columns.str.strip().str.upper()
        tabla_horario_estacional.columns = tabla_horario_estacional.columns.str.replace('.', '')
        tabla_horario_estacional.columns = tabla_horario_estacional.columns.str.replace(' ', '_').map(unidecode)
        tabla_horario_estacional_01 = tabla_horario_estacional[['HORARIO', 'HORARIO_(+1)']].copy()
        return tabla_horario_estacional_01

    @staticmethod
    def filter_prog_acad_icpna(df_prog_acad: pd.DataFrame):
        """
        Docstring for filter_prog_acad_icpna
        
        :param df_prog_acad: Description
        :type df_prog_acad: pd.DataFrame
        
        Filtros aplicados para una venta regular
        
        """
        es_cancelado = df_prog_acad['ESTADO'] == 'Cur. Cancelado'
        es_vecor = df_prog_acad['SEDE'] == 'VECOR'
        es_corporate = df_prog_acad['NIVEL'] == 'Corporate'
        end_with_pv = df_prog_acad['CODIGO_DE_CURSO'].str.contains('.+[pP][Vv]$')
        end_with_p = df_prog_acad['CODIGO_DE_CURSO'].str.contains('.+[pP]$')
        end_with_hd = df_prog_acad['CODIGO_DE_CURSO'].str.contains('.+[Hh][Dd]$')
        break_periodo = df_prog_acad['PERIODO'] < 202409

        return (
            df_prog_acad[(~(es_cancelado | es_vecor |end_with_hd | ((end_with_pv | end_with_p) & break_periodo) |
                            (end_with_pv & ~break_periodo) | es_corporate))].copy())
    
    @staticmethod
    def filter_model(df_prog_acad: pd.DataFrame):
        es_satelite = df_prog_acad['SEDE'].isin(
            ['San Juan de Lurigancho Satélite', 'San Juan de Miraflores Satélite'])

        return df_prog_acad[~es_satelite].copy()

    def fetch_prog_acad_by_periodo(self, periodo: int):
        assert isinstance(periodo, int)
        year = periodo // 100
        # print(os.getcwd())
        path = self.input_datastore / 'prog-acad' / str(year) / f'{periodo}.xlsx'
        prog_acad = pd.read_excel(path, sheet_name='Sheet1', skiprows=1)
        prog_acad.columns = prog_acad.columns.astype('string')
        prog_acad.columns = prog_acad.columns.str.strip().str.upper()
        prog_acad.columns = prog_acad.columns.str.replace('.', '')
        prog_acad.columns = prog_acad.columns.str.replace(' ', '_').map(unidecode)

        prog_acad['PERIODO'] = prog_acad['PERIODO'].astype('int32')
        prog_acad['SEDE'] = prog_acad['SEDE'].astype('string')
        prog_acad['MODALIDAD'] = prog_acad['MODALIDAD'].astype('string')
        prog_acad['FASE'] = prog_acad['FASE'].astype('string')
        prog_acad['NIVEL'] = prog_acad['NIVEL'].astype('string')
        prog_acad['CODIGO_DE_CURSO'] = prog_acad['CODIGO_DE_CURSO'].astype('string')
        prog_acad['DESCRIPCION'] = prog_acad['DESCRIPCION'].astype('string')
        prog_acad['FRECUENCIA'] = prog_acad['FRECUENCIA'].astype('string')
        prog_acad['INTENSIDAD'] = prog_acad['INTENSIDAD'].astype('string')
        prog_acad['HORARIO'] = prog_acad['HORARIO'].astype('string')
        prog_acad['AULA'] = prog_acad['AULA'].astype('string')
        prog_acad['COD_DOCENTE'] = prog_acad['COD_DOCENTE'].astype('string')
        prog_acad['PROFESOR'] = prog_acad['PROFESOR'].astype('string')
        prog_acad['CANT_MATRICULADOS'] = prog_acad['CANT_MATRICULADOS'].astype('int32')
        prog_acad['VACANTES_USADAS'] = prog_acad['VACANTES_USADAS'].astype('int32')
        prog_acad['VACANTES_DISP'] = prog_acad['VACANTES_DISP'].astype('int32')
        prog_acad['VAC_HABILITADAS'] = prog_acad['VAC_HABILITADAS'].astype('int32')
        prog_acad['AFORO'] = prog_acad['AFORO'].astype('float64')
        prog_acad['ESTADO'] = prog_acad['ESTADO'].astype('string')
        prog_acad['FECHA_CREACION'] = prog_acad['FECHA_CREACION'].astype('string')
        n_rows = int(prog_acad.shape[0])

        print(f"Periodo:{periodo} || N row:{n_rows:,}")
        # Solo utilizar cuando existe un error
        # prog_acad = self.filter_prog_acad(prog_acad)
        # print(f"    ¦- N rows:{n_rows:,}", f"¦- N rows after filtering:{prog_acad.shape[0]:,}",
        #       f"¦- N rows filtered:{(n_rows - prog_acad.shape[0]):,}")
        return prog_acad

    def fetch_prog_acad(self):

        collection = []
        for periodo in self.periodos:
            prog_acad = self.fetch_prog_acad_by_periodo(periodo)
            collection.append(prog_acad)
        prog_acad_consol = pd.concat(collection, ignore_index=True)

        return prog_acad_consol

    def pull_tabla_curso_actual(self, df_curso_actual: pd.DataFrame):
        path = self.output_datastore /'dim'/ self.platinum_version/'cursos_actual.parquet'
        df_curso_actual.to_parquet(path, index=False)

    def pull_tabla_turno_disponible(self, df_turno_disponible: pd.DataFrame):
        path = self.output_datastore /'dim'/ self.platinum_version/'turno_disponible.parquet'
        df_turno_disponible.to_parquet(path, index=False)

    def pull_tabla_vac_estandar(self, df_tabla_vac_estandar: pd.DataFrame):
        path = self.output_datastore /'dim'/ self.platinum_version/'tabla_vac_estandar.parquet'
        df_tabla_vac_estandar.to_parquet(path, index=False)

    def pull_tabla_pe(self, df_tabla_pe: pd.DataFrame):
        path = self.output_datastore /'dim'/ self.platinum_version/'tabla_pe.parquet'
        df_tabla_pe.to_parquet(path, index=False)

    def pull_tabla_horario(self, df_tabla_horario: pd.DataFrame):
        path = self.output_datastore /'dim'/ self.platinum_version/'tabla_horario.parquet'
        df_tabla_horario.to_parquet(path, index=False)

    def pull_tabla_curso_diario_to_sabatino(
            self, df_tabla_curso_diario_to_sabatino: pd.DataFrame):
        path = self.output_datastore /'dim'/ self.platinum_version/'tabla_curso_diario_to_sabatino.parquet'
        df_tabla_curso_diario_to_sabatino.to_parquet(path, index=False)

    def pull_tabla_horario_diario_to_sabatino(
            self,
            df_tabla_horario_diario_to_sabatino: pd.DataFrame):
        path = self.output_datastore /'dim'/ self.platinum_version/'tabla_horario_diario_to_sabatino.parquet'
        df_tabla_horario_diario_to_sabatino.to_parquet(path, index=False)

    def pull_synthetic(self, df_synthetic: pd.DataFrame):
        path = self.output_datastore / 'synthetic'/ self.platinum_version/'synthetic.parquet'
        df_synthetic.to_parquet(path, index=False)

    def pull_tabla_curso_inicial(self, df_tabla_curso_inicial: pd.DataFrame):
        path = self.output_datastore / 'dim'/self.platinum_version/'tabla_curso_inicial.parquet'
        df_tabla_curso_inicial.to_parquet(path, index=False)

    def pull_tabla_curso_estacional(self, df_tabla_curso_estacional: pd.DataFrame):
        path = self.output_datastore / 'dim'/self.platinum_version/'tabla_curso_estacional.parquet'
        df_tabla_curso_estacional.to_parquet(path, index=False)

    def pull_tabla_curso_acumulado(self, df_tabla_curso_acumulado: pd.DataFrame):
        path = self.output_datastore / 'dim'/self.platinum_version/'tabla_curso_acumulado.parquet'
        df_tabla_curso_acumulado.to_parquet(path, index=False)

    def pull_prog_acad(self, df_prog_acad: pd.DataFrame):

        for idx, prog_acad in df_prog_acad.groupby('PERIODO'):
            assert isinstance(idx, int)
            periodo = idx
            year = periodo // 100
            if not (self.output_datastore/'prog-acad'/self.platinum_version/f'{year}').exists():
                (self.output_datastore/'prog-acad'/self.platinum_version/f'{year}').mkdir()
                
            path = self.output_datastore / 'prog-acad' / self.platinum_version / f'{year}'/f'{periodo}.parquet'
            
            prog_acad.to_parquet(path, index=False)

    def fetch_and_transform_and_pull(self):
        
        # ---------------------------------------------------------------------
        # --------------------------- Fetch -----------------------------------
        # ---------------------------------------------------------------------

        # if not (self.output_datastore /self.raw_version/ 'dim').exists():
        #     (self.output_datastore /self.raw_version/ 'dim').mkdir(parents=True, exist_ok=True)

        # if not (self.output_datastore /self.raw_version/ 'prog-acad').exists():
        #     (self.output_datastore /self.raw_version/ 'prog-acad').mkdir(parents=True, exist_ok=True)

        # if not (self.output_datastore /self.raw_version/ 'synthetic').exists():
        #     (self.output_datastore /self.raw_version/ 'synthetic').mkdir(parents=True, exist_ok=True)
        

        tabla_horario = self.fetch_tabla_horario()
        tabla_curso_actual = self.fetch_tabla_curso_actual()
        tabla_curso_acumulado = self.fetch_tabla_curso_acumulado()
        tabla_curso_inicial = self.fetch_tabla_curso_inicial()
        tabla_curso_estacional = self.fetch_tabla_curso_estacional()
        tabla_turno_disponible = self.fetch_tabla_turno_disponible()
        tabla_curso_diario_to_sabatino = self.fetch_tabla_curso_diario_to_sabatino()
        tabla_horario_diario_to_sabatino = self.fetch_tabla_horario_diario_to_sabatino()
        tabla_vac_estandar = self.fetch_tabla_vac_estandar()
        tabla_pe = self.fetch_tabla_pe()
        synthetic = self.fetch_synthetic()
        prog_acad = self.fetch_prog_acad()
        # n_rows = prog_acad.shape[0]
        prog_acad = self.filter_prog_acad_icpna(prog_acad)
        
        prog_acad = self.filter_model(prog_acad)
        # print(f"    ¦- N rows:{n_rows:,}", f"¦- N rows after filtering:{prog_acad.shape[0]:,}",
        #       f"¦- N rows filtered:{(n_rows - prog_acad.shape[0]):,}")
        # tabla_aula = self.fetch_tabla_aula()
        # tabla_sede = self.fetch_tabla_sede()

        curso_acumulado = tabla_curso_acumulado[
            ['CURSO_ANTERIOR', 'CURSO_ACTUAL']].rename(
            columns={'CURSO_ANTERIOR': 'CODIGO_DE_CURSO'}).copy()

        horario = tabla_horario[['HORARIO', 'HORARIO_ACTUAL']].copy()

        # ---------------------------------------------------------------------
        # -------------------------- Transform --------------------------------
        # ---------------------------------------------------------------------
        prog_acad['SEDE'] = prog_acad['SEDE'].str.replace('Provincias - ', '')

        prog_acad['SEDE'] = np.where(
            prog_acad['SEDE'].isin(['Lima Norte Satélite 2', 'Lima Norte']),
            'Lima Norte Satélite',
            prog_acad['SEDE']
        )

        # join
        prog_acad_01 = prog_acad.merge(
            curso_acumulado,
            how='left',
            on=['CODIGO_DE_CURSO']
        )

        any_null_on_curso_actual = prog_acad_01['CURSO_ACTUAL'].isnull()
        zero_null_on_curso_actual = np.sum(any_null_on_curso_actual) == 0

        if not zero_null_on_curso_actual:
            prog_acad_null_curso_actual = prog_acad_01.loc[
                any_null_on_curso_actual, 
                ['PERIODO', 'SEDE', 'CODIGO_DE_CURSO','HORARIO']].copy()
            print("NULL on Curso Actual in Prog Acad", prog_acad_null_curso_actual)           
        
        assert prog_acad_01['CURSO_ACTUAL'].isnull().sum() == 0

        prog_acad_02 = prog_acad_01.merge(
            horario,
            how='left',
            on=['HORARIO']
        )

        any_null_on_horario_actual = prog_acad_02['HORARIO_ACTUAL'].isnull()
        zero_null_on_horario_actual = np.sum(any_null_on_horario_actual) == 0

        if not zero_null_on_horario_actual:
            prog_acad_null_horario_actual = prog_acad_02.loc[
                any_null_on_horario_actual, 
                ['PERIODO', 'SEDE', 'CODIGO_DE_CURSO','HORARIO', 'HORARIO_ACTUAL']].copy()
            print("NULL on Curso Actual in Prog Acad", prog_acad_null_horario_actual)     

        assert prog_acad_02['HORARIO_ACTUAL'].isnull().sum() == 0

        prog_acad_03 = prog_acad_02.groupby(
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL'], as_index=False
        ).agg(
            CANT_CLASES=pd.NamedAgg(
                column='CURSO_ACTUAL', aggfunc='size'
            ),
            CANT_ALUMNOS=pd.NamedAgg(
                column='CANT_MATRICULADOS', aggfunc='sum'
            )
        )

        # join in synthetic
        synthetic_01 = synthetic.merge(
            curso_acumulado,
            how='left',
            on=['CODIGO_DE_CURSO']
        )

        any_null_on_curso_actual = synthetic_01['CURSO_ACTUAL'].isnull()
        zero_null_on_curso_actual = np.sum(any_null_on_curso_actual) == 0

        if not zero_null_on_curso_actual:
            prog_acad_null_curso_actual = synthetic_01.loc[
                any_null_on_curso_actual, 
                ['PERIODO', 'SEDE', 'CODIGO_DE_CURSO','HORARIO']].copy()
            print("NULL on Curso Actual in Prog Acad", prog_acad_null_curso_actual) 
        
        assert synthetic_01['CURSO_ACTUAL'].isnull().sum() == 0
        
        synthetic_02 = synthetic_01.merge(
            horario,
            how='left',
            on=['HORARIO']
        )

        any_null_on_horario_actual = synthetic_02['HORARIO_ACTUAL'].isnull()
        zero_null_on_horario_actual = np.sum(any_null_on_horario_actual) == 0

        if not zero_null_on_horario_actual:
            prog_acad_null_horario_actual = synthetic_02.loc[
                any_null_on_horario_actual, 
                ['PERIODO', 'SEDE', 'CODIGO_DE_CURSO','HORARIO', 'HORARIO_ACTUAL']].copy()
            print("NULL on Curso Actual in Prog Acad", prog_acad_null_horario_actual)     

        
        assert synthetic_02['HORARIO_ACTUAL'].isnull().sum() == 0
        
        synthetic_03 = synthetic_02.groupby(
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL'], as_index=False
        ).agg(
            CANT_CLASES=pd.NamedAgg(
                column='CANT_CLASES', aggfunc='sum'
            ),
            CANT_ALUMNOS=pd.NamedAgg(
                column='CANT_ALUMNOS', aggfunc='sum'
            )
        )
        
        # ---------------------------------------------------------------------
        # --------------------------- Pull ---------------------------------
        # ---------------------------------------------------------------------
        if not (self.output_datastore / 'dim'/self.platinum_version).exists():
            (self.output_datastore / 'dim'/self.platinum_version).mkdir(parents=True, exist_ok=True)

        if not (self.output_datastore / 'prog-acad'/self.platinum_version).exists():
            (self.output_datastore / 'prog-acad'/self.platinum_version).mkdir(parents=True, exist_ok=True)

        if not (self.output_datastore / 'synthetic'/self.platinum_version).exists():
            (self.output_datastore / 'synthetic'/self.platinum_version).mkdir(parents=True, exist_ok=True)
        
        self.pull_prog_acad(prog_acad_03)
        self.pull_synthetic(synthetic_03)
        self.pull_tabla_turno_disponible(tabla_turno_disponible)
        self.pull_tabla_curso_actual(tabla_curso_actual)
        self.pull_tabla_curso_acumulado(tabla_curso_acumulado)
        self.pull_tabla_vac_estandar(tabla_vac_estandar)
        self.pull_tabla_curso_inicial(tabla_curso_inicial)
        self.pull_tabla_curso_estacional(tabla_curso_estacional)
        self.pull_tabla_curso_diario_to_sabatino(tabla_curso_diario_to_sabatino)
        self.pull_tabla_horario_diario_to_sabatino(tabla_horario_diario_to_sabatino)
        self.pull_tabla_pe(tabla_pe)
        self.pull_tabla_horario(tabla_horario)

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_datastore", dest='input_datastore',
                        type=str)
    parser.add_argument("--output_datastore", dest='output_datastore',
                        type=str)
    parser.add_argument("--ult_periodo", dest='ult_periodo',
                        type=int)
    parser.add_argument("--n_periodos", dest='n_periodos',
                        type=str, help='Es un entero positivo o "None" (si all_periodos="True"). Por ejemplo: "1", "2" o "3", ...')
    parser.add_argument("--all_periodos", dest='all_periodos',
                        type=str, help='Es "True" o "False"')
    parser.add_argument("--platinum_version", dest='platinum_version',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

def main(args):
    input_datastore=args.input_datastore
    output_datastore=args.output_datastore
    platinum_version=args.platinum_version
    ult_periodo=args.ult_periodo
    n_periodos=eval(args.n_periodos)
    all_periodos=eval(args.all_periodos)
    
    etl = Etl(
        input_datastore, 
        output_datastore, 
        platinum_version,
        ult_periodo, 
        n_periodos, 
        all_periodos)
    
    etl.fetch_and_transform_and_pull()

if __name__ == '__main__':
    # input_datastore =  base_data
    # raw_version = "v1"
    # output_datastore = ml_data
    # platinum_version = "v1"
    # python src/data/etl.py --input_datastore './data/base_data/rawdata/'  --output_datastore  './data/ml_data/platinumdata'  --ult_periodo 202601  --n_periodos "None"  --all_periodos "True" --platinum_version "v1"
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    # print(args)
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
