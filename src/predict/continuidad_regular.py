import pandas as pd
# from sklearn.ensemble import RandomForestRegressor # type: ignore
# from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor # type: ignore
# from utils import calculate_classes, join_target, calculate_metrics, calculate_alumnos
import numpy as np
from catboost import CatBoostRegressor, Pool
from pathlib import Path
import argparse
from utils_predict import get_mapping_tipos, get_ahead_n_periodos 
from utils_metrics import calculate_classes , calculate_metrics, fac_to_cant
# import mlflow


class TrainContinuidad:
    def __init__(self,
                 input_model_datastore: str,
                 input_feats_datastore: str,
                 output_predict_datastore: str,
                 feats_version: str,
                 model_version: str,
                 tipo: str,
                 ):
        
        self.input_feats_datastore = Path(input_feats_datastore)
        self.input_model_datastore = Path(input_model_datastore)
        self.output_predict_datastore = Path(output_predict_datastore)
        self.tipo  = tipo
        self.feats_version = feats_version
        self.model_version = model_version

    def get_data_test(self, periodo: int):
        data_model_test = pd.read_parquet(
            self.input_feats_datastore
            / "test"
            / self.feats_version
            / f"data_feats_{self.tipo}_{periodo}.parquet"
        )
        return data_model_test

    def load_model(self, model_periodo:int):
        print(f"Cargando modelo desde: {self.input_model_datastore}")
        model = CatBoostRegressor()

        input_model_datastore = self.input_model_datastore
        
        model.load_model(self.input_model_datastore/'test'/self.model_version/f"{self.tipo}_{model_periodo}.cbm")
        # model = mlflow.sklearn.load_model(
        #     self.input_model_datastore
        #     # f"models:/{self.tipo}_{periodo}@dev"
        # )
        return model
    

    def get_data_predict(self, periodo:int, model):
        """
        Compute metrics for the forecast.
        
        Args:
            df_forecast (pd.DataFrame): DataFrame with forecast data.
        
        Returns:
            pd.DataFrame: DataFrame with forecast data after computing metrics.
        """
        # read data model
        data_model_eval = self.get_data_test(periodo)
        
        # ----------------------------
        
        meses = [1, 2, 3]

        base = ['PERIODO_TARGET', 'CURSO_ACTUAL', 'IDX', 'PE', 'VAC_ACAD_ESTANDAR']

        cat_features = ['LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']

        if periodo % 100 in meses:
            # num_features = ['CANT_ALUMNOS_LAG_12', 'CANT_ALUMNOS_ANTERIOR', 
            #                  'DIFF_ALUMNOS_SPLY', 'PCT_ALUMNOS_SPLY']
            
            # x = num_features + cat_features
            # cat_features = ['LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']
            if periodo%100 in [1, 3]:
                num_features = [
                    'CANT_ALUMNOS_LAG_12', 
                    'CANT_ALUMNOS_ANTERIOR', 
                    # 'DIFF_ALUMNOS_SPLY', 
                    'PCT_ALUMNOS_SPLY'
                    ]
                x = num_features + cat_features
                target = "PCT_ALUMNOS"
                # predict = f"{target}_PREDICT"
                data_model_eval = data_model_eval[base + num_features + cat_features +
                                                  ['CANT_CLASES_ANTERIOR']].copy()
                X_eval = data_model_eval[x].copy()
                preds = model.predict(X_eval)
                preds = np.asarray(data_model_eval ['CANT_ALUMNOS_ANTERIOR'] * (1 + preds))
                preds = np.where(preds < 0, 0, preds)
                preds = preds // 1 + np.where(preds %1 >= 0.4,1, 0)
                data_model_eval[f'CANT_ALUMNOS_PREDICT'] = preds

                # predict = f"{target}_PREDICT"
            else:
                num_features = [
                    # 'CANT_ALUMNOS_LAG_12', 
                    'CANT_ALUMNOS_ANTERIOR', 
                    # 'DIFF_ALUMNOS_SPLY', 
                    # 'PCT_ALUMNOS_SPLY'
                    ]
                x = num_features + cat_features
                target = "CANT_ALUMNOS"
                predict = f"{target}_PREDICT"
                data_model_eval = data_model_eval[base + num_features + cat_features + 
                                                  ['CANT_CLASES_ANTERIOR']].copy()
                X_eval = data_model_eval[x].copy()
                preds = model.predict(X_eval)

                preds = np.where(preds < 0, 0, preds)
                preds = preds // 1 + np.where(preds %1 >= 0.4,1, 0)
                
                data_model_eval[predict] = preds

                # predict = f"{target}_PREDICT"
            
        else:
            num_features = ['CANT_ALUMNOS_ANTERIOR']
            x = num_features + cat_features
            target = "CANT_ALUMNOS"
            predict = f"{target}_PREDICT"

            data_model_eval = data_model_eval[base + num_features + cat_features + ['CANT_CLASES_ANTERIOR']].copy()
            X_eval = data_model_eval[x].copy()
            preds = model.predict(X_eval)

            preds = np.where(preds < 0, 0, preds)
            preds = preds // 1 + np.where(preds %1 >= 0.4,1, 0)
            
            data_model_eval[predict] = preds

        # data_model_eval.to_excel('data_model_eval.xlsx', index=False)
        # on_cols = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL']

        data_model_eval = calculate_classes(data_model_eval)
        
        return data_model_eval
    
    def upload_data_predict(
        self, model_version: str, model_periodo: int, periodo: int, 
        df_model_predict: pd.DataFrame, mode:str):
      
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_datastore", dest="input_model_datastore", type=str)
    parser.add_argument("--input_feats_datastore", dest="input_feats_datastore", type=str)
    parser.add_argument("--output_predict_datastore", dest="output_predict_datastore", type=str)
    parser.add_argument("--feats_version", dest="feats_version", type=str)
    parser.add_argument("--n_eval_periodos", dest="n_eval_periodos", type=int, default=-1)
    parser.add_argument("--model_periodo", dest="model_periodo", type=int)
    parser.add_argument("--model_version", dest="model_version", type=str)
    parser.add_argument("--periodo", dest="periodo", type=int, default=-1)
    parser.add_argument("--mode", dest="mode", type=str)
    parser.add_argument("--with_tipo", dest="with_tipo", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args):
    input_model_datastore = args.input_model_datastore
    input_feats_datastore = args.input_feats_datastore
    
    output_predict_datastore = args.output_predict_datastore
    
    feats_version = args.feats_version
    model_periodo = args.model_periodo
    model_version = args.model_version
    n_eval_periodos = args.n_eval_periodos

    periodo = args.periodo
    mode = args.mode
    with_tipo = args.with_tipo

    tipo = 'continuidad_regular'
    eval_tipo =eval(with_tipo)
    
    if not eval_tipo:
        input_model_datastore = f"{input_model_datastore}/{tipo}"
        input_feats_datastore = f"{input_feats_datastore}/{tipo}"
        output_predict_datastore = f"{output_predict_datastore}/{tipo}"
    
    train_continuidad = TrainContinuidad(
        input_model_datastore,
        input_feats_datastore,
        output_predict_datastore,
        feats_version,
        model_version,
        tipo
        )
    
    model = train_continuidad.load_model(model_periodo)

    if periodo == -1:
        assert n_eval_periodos >= 1
        periodos = get_ahead_n_periodos(model_periodo, n_eval_periodos)
    else:
        periodos = [periodo]

    for periodo in periodos:
        mapping_tipos = get_mapping_tipos(periodo)
        if mapping_tipos[train_continuidad.tipo]:
            df_model_predict = train_continuidad.get_data_predict(periodo, model)
            train_continuidad.upload_data_predict(
                model_version, model_periodo, periodo, df_model_predict, mode)
    

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