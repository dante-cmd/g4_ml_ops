"""
Script para predecir la continuidad regular horario
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from pathlib import Path
from utils_metrics import calculate_classes , calculate_metrics, fac_to_cant
from utils_predict import get_mapping_tipos, get_ahead_n_periodos
import argparse 


class TrainContinuidadToHorario:
    """
    Clase para entrenar el modelo de continuidad regular horario.
    """
    def __init__(self,
                 input_feats_datastore:str,
                 input_predict_datastore:str,
                 input_model_datastore:str,
                 output_predict_datastore:str,
                 feats_version:str,
                 model_version:str,
                 tipo:str,
                 tipo_continuidad:str
                 ):
        
        self.input_feats_datastore = Path(input_feats_datastore)
        self.input_predict_datastore = Path(input_predict_datastore)
        self.output_predict_datastore = Path(output_predict_datastore)
        self.input_model_datastore = Path(input_model_datastore)
        self.tipo = tipo
        self.tipo_continuidad = tipo_continuidad
        self.feats_version = feats_version
        self.model_version = model_version
    
    def get_data_test(self, periodo: int):
        """
        Obtiene los datos de test para el periodo dado.
        """
        data_model_test = pd.read_parquet(
            self.input_feats_datastore
            / "test"
            / self.feats_version
            / f"data_feats_{self.tipo}_{periodo}.parquet"
        )
        return data_model_test

    def load_model(self, model_periodo:int):
        """
        Carga el modelo entrenado.
        """
        print(f"Cargando modelo desde: {self.input_model_datastore}")
        model = CatBoostRegressor()
        
        model.load_model(self.input_model_datastore/'test'/self.model_version/f"{self.tipo}_{model_periodo}.cbm")
        # model = mlflow.sklearn.load_model(
        #     self.input_model_datastore
        #     # f"models:/{self.tipo}_{periodo}@dev"
        # )
        return model
    
    def get_data_input_predict(self, model_periodo:int, periodo:int):
        """
        Obtiene los datos de input para el periodo dado.
        """
        data_model_test = pd.read_parquet(
            self.input_predict_datastore/"test"/f"data_predict_{self.model_version}_{model_periodo}_{self.tipo_continuidad}_{periodo}.parquet")
        return data_model_test    

    def get_data_predict(self, model_periodo:int ,periodo:int, model):
        """
        Compute metrics for the forecast.
        
        Args:
            df_forecast (pd.DataFrame): DataFrame with forecast data.
        
        Returns:
            pd.DataFrame: DataFrame with forecast data after computing metrics.
        """
        
        # Load logging
        # log_metrics = self.get_logging()

        # Load model
        # model = CatBoostRegressor()
        # model.load_model(self.input_model_datastore/f"{self.tipo}_{self.periodo_model}.cbm")

        # read data model
        data_model_eval = self.get_data_test(periodo)
        df_predict = self.get_data_input_predict(model_periodo, periodo)
        # data_model_eval = pd.read_parquet(
        #     self.input_feats_test_datastore/f'data_feats_test_{self.tipo}_{periodo}.parquet')
        # target = 'CANT_ALUMNOS'
        # ----------------------------
        
        meses = [1, 2, 3]

        base = ['PERIODO_TARGET', 'CURSO_ACTUAL','HORARIO_ACTUAL',
                'IDX', 'PE', 'VAC_ACAD_ESTANDAR']

        # cat_features = ['LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']
        cat_features = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']
        
        target = 'FAC_ALUMNOS'
        predict = f"{target}_PREDICT"

        if periodo % 100 in meses:
            num_features = ['FAC_ALUMNOS_LAG_12', 'FAC_ALUMNOS_ANTERIOR']
            # num_features = ['CANT_ALUMNOS_LAG_12', 'CANT_ALUMNOS_ANTERIOR', 
            #                  'DIFF_ALUMNOS_SPLY', 'PCT_ALUMNOS_SPLY']
            x = num_features + cat_features

            data_model_eval = data_model_eval[base + num_features + cat_features + 
                                              ['CANT_CLASES_ANTERIOR', 'CANT_ALUMNOS_ANTERIOR']].copy()
            X_eval = data_model_eval[x].copy()
            preds = model.predict(X_eval)

            preds = np.where(preds < 0, 0, preds)
            # preds = preds // 1 + np.where(preds %1 >= 0.4,1, 0)
                
            data_model_eval[predict] = preds
            
        else:
            num_features = ['FAC_ALUMNOS_ANTERIOR']
            x = num_features + cat_features
            # target = "CANT_ALUMNOS"
            predict = f"{target}_PREDICT"

            data_model_eval = data_model_eval[base + num_features + cat_features + ['CANT_CLASES_ANTERIOR', 'CANT_ALUMNOS_ANTERIOR']].copy()
            X_eval = data_model_eval[x].copy()
            preds = model.predict(X_eval)

            preds = np.where(preds < 0, 0, preds)
            # preds = preds // 1 + np.where(preds %1 >= 0.4,1, 0)
            
            data_model_eval[predict] = preds

        # data_model_eval.to_excel('data_model_eval_horario.xlsx', index=False)

        # on_cols = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']
        data_model_eval = fac_to_cant(data_model_eval, df_predict)

        data_model_eval = calculate_classes(data_model_eval)

        return data_model_eval


        return model
    
    def upload_data_predict(
        self, model_version: str, model_periodo: int, periodo: int, 
        df_model_predict: pd.DataFrame, mode:str):
        """
        Sube los datos de predicción al datastore.
        """
        
        path_model_version = self.output_predict_datastore/"test"
            
        path_model_version.mkdir(parents=True, exist_ok=True)
        
        df_model_predict.to_parquet(
            path_model_version / f"data_predict_{model_version}_{model_periodo}_{self.tipo}_{periodo}.parquet"
        )
        
        # path_model_version = self.output_predict_datastore / "test" 
        assert mode in ["dev", "prod"]
        
        if mode == "dev":
            df_model_predict.to_parquet(
            path_model_version / f"data_predict_dev_{model_periodo}_{self.tipo}_{periodo}.parquet"
        )
        else:
            df_model_predict.to_parquet(
            path_model_version / f"data_predict_prod_{model_periodo}_{self.tipo}_{periodo}.parquet"
        )


def parse_args():
    """
    Parsea los argumentos de la línea de comandos.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_datastore", dest="input_model_datastore", type=str)
    parser.add_argument("--input_feats_datastore", dest="input_feats_datastore", type=str)
    parser.add_argument("--input_predict_datastore", dest="input_predict_datastore", type=str)
    parser.add_argument("--output_predict_datastore", dest="output_predict_datastore", type=str)
    parser.add_argument("--feats_version", dest="feats_version", type=str)
    parser.add_argument("--n_eval_periodos", dest="n_eval_periodos", type=int, default=-1)
    parser.add_argument("--model_periodo", dest="model_periodo", type=int)
    parser.add_argument("--model_version", dest="model_version", type=str)
    parser.add_argument("--periodo", dest="periodo", type=int, default=-1)
    parser.add_argument("--mode", dest="mode", type=str)
    parser.add_argument("--with_tipo", dest="with_tipo", type=str)

    args = parser.parse_args()

    return args


def main(args):
    """
    Función principal para entrenar el modelo de continuidad regular horario.
    """
    input_feats_datastore = args.input_feats_datastore
    input_predict_datastore = args.input_predict_datastore
    input_model_datastore = args.input_model_datastore

    output_predict_datastore = args.output_predict_datastore

    feats_version = args.feats_version
    model_periodo = args.model_periodo
    model_version = args.model_version
    n_eval_periodos = args.n_eval_periodos

    periodo = args.periodo
    mode = args.mode
    with_tipo = args.with_tipo

    tipo = 'continuidad_regular_horario'
    tipo_continuidad = 'continuidad_regular'

    eval_tipo = eval(with_tipo)
    if not eval_tipo:
        input_feats_datastore = f"{input_feats_datastore}/{tipo}"
        input_predict_datastore = f"{input_predict_datastore}/{tipo_continuidad}"
        input_model_datastore = f"{input_model_datastore}/{tipo}"
        output_predict_datastore = f"{output_predict_datastore}/{tipo}"
    
    train_continuidad_horario = TrainContinuidadToHorario(
        input_feats_datastore,
        input_predict_datastore,
        input_model_datastore,
        output_predict_datastore,
        feats_version,
        model_version,
        tipo,
        tipo_continuidad)
    
    model = train_continuidad_horario.load_model(model_periodo)

    if periodo == -1:
        assert n_eval_periodos >= 1
        periodos = get_ahead_n_periodos(model_periodo, n_eval_periodos)
    else:
        periodos = [periodo]

    for periodo in periodos:
        mapping_tipos = get_mapping_tipos(periodo)
        
        if mapping_tipos[train_continuidad_horario.tipo]:
            df_model_predict = train_continuidad_horario.get_data_predict(model_periodo,periodo, model)
            train_continuidad_horario.upload_data_predict(model_version, model_periodo, periodo, df_model_predict, mode)
    
    

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