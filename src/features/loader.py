import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from unidecode import unidecode
import os
import argparse
from pathlib import Path
from utils_loader import get_all_periodos, validate_periodos, parse_args


class Loader:

    def __init__(self, 
                 input_datastore: str, 
                 platinum_version: str,
                 ult_periodo: int
                 ) -> None:
        self.input_datastore = Path(input_datastore)
        self.platinum_version = platinum_version
        self.ult_periodo = ult_periodo
        # self.n_periodos = n_periodos
        # self.all_periodos = all_periodos
        # self.periodos = validate_periodos(n_periodos, ult_periodo, all_periodos)
    
    def fetch_tabla_curso_actual(self):
        path =  self.input_datastore /'dim'/self.platinum_version/'cursos_actual.parquet'
        df_curso_actual = pd.read_parquet(path)
        return df_curso_actual

    def fetch_tabla_curso_acumulado(self):
        path =  self.input_datastore /'dim'/self.platinum_version/'tabla_curso_acumulado.parquet'
        df_curso_acumulado = pd.read_parquet(path)
        return df_curso_acumulado

    def fetch_tabla_turno_disponible(self):
        path =  self.input_datastore /'dim'/self.platinum_version/'turno_disponible.parquet'
        df_turno_disponible = pd.read_parquet(path)
        return df_turno_disponible

    def fetch_tabla_vac_estandar(self):
        path =  self.input_datastore /'dim'/self.platinum_version/'tabla_vac_estandar.parquet'
        df_tabla_vac_estandar = pd.read_parquet(path)
        return df_tabla_vac_estandar

    def fetch_tabla_pe(self):
        path =  self.input_datastore /'dim'/self.platinum_version/'tabla_pe.parquet'
        tabla_pe = pd.read_parquet(path)
        return tabla_pe

    def fetch_tabla_horario(self):
        path =  self.input_datastore /'dim'/self.platinum_version/'tabla_horario.parquet'
        tabla_horario = pd.read_parquet(path)
        return tabla_horario
    
    def fetch_tabla_curso_diario_to_sabatino(self):
        path = (
                self.input_datastore /
                'dim'/self.platinum_version/'tabla_curso_diario_to_sabatino.parquet')
        tabla_pe = pd.read_parquet(path)
        return tabla_pe

    def fetch_tabla_horario_diario_to_sabatino(self):
        path = (
                self.input_datastore /
                'dim'/self.platinum_version/'tabla_horario_diario_to_sabatino.parquet')
        tabla_pe = pd.read_parquet(path)
        return tabla_pe

    def fetch_synthetic(self):
        path =  self.input_datastore /'synthetic'/self.platinum_version/'synthetic.parquet'
        synthetic = pd.read_parquet(path)
        return synthetic

    def fetch_tabla_curso_inicial(self):
        path =  self.input_datastore /'dim'/self.platinum_version/'tabla_curso_inicial.parquet'
        df_tabla_curso_inicial = pd.read_parquet(path)
        return df_tabla_curso_inicial

    def fetch_prog_acad(self):

        collection = []
        periodos = get_all_periodos(self.ult_periodo)
        for periodo in periodos:
            year = periodo // 100
            path = self.input_datastore / f'prog-acad'/self.platinum_version/f'{year}/{periodo}.parquet'
            try:
                prog_acad = pd.read_parquet(path)
                collection.append(prog_acad)
            except Exception as e:
                print(f"⚠️ No existe {path}")
                print(e)

        prog_acad_consol = pd.concat(collection, ignore_index=True)
        return prog_acad_consol
    
    def fetch_all(self) -> dict:
        tablas = {}
        tabla_curso_actual = self.fetch_tabla_curso_actual()
        tabla_curso_acumulado = self.fetch_tabla_curso_acumulado()
        tabla_curso_inicial = self.fetch_tabla_curso_inicial()
        tabla_pe = self.fetch_tabla_pe()
        tabla_horario = self.fetch_tabla_horario()
        tabla_vac_estandar = self.fetch_tabla_vac_estandar()
        tabla_horario_diario_to_sabatino = self.fetch_tabla_horario_diario_to_sabatino()
        tabla_curso_diario_to_sabatino = self.fetch_tabla_curso_diario_to_sabatino()
        tabla_turno_disponible = self.fetch_tabla_turno_disponible()
        prog_acad = self.fetch_prog_acad()
        synthetic = self.fetch_synthetic()

        tablas['tabla_curso_actual']=tabla_curso_actual
        tablas['tabla_curso_acumulado']=tabla_curso_acumulado
        tablas['tabla_curso_inicial']=tabla_curso_inicial
        tablas['tabla_pe']=tabla_pe
        tablas['tabla_horario']=tabla_horario
        tablas['tabla_vac_estandar']=tabla_vac_estandar
        tablas['tabla_horario_diario_to_sabatino']=tabla_horario_diario_to_sabatino
        tablas['tabla_curso_diario_to_sabatino']=tabla_curso_diario_to_sabatino
        tablas['tabla_turno_disponible']=tabla_turno_disponible
        tablas['prog_acad']=prog_acad
        tablas['synthetic']=synthetic

        return tablas


def main(args):
    input_datastore=args.input_datastore
    ult_periodo=args.ult_periodo
    platinum_version=args.platinum_version
    
    loader = Loader(
        input_datastore, 
        platinum_version,
        ult_periodo)
    
    loader.fetch_all()


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