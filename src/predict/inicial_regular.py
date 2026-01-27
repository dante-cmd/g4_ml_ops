import pandas as pd
# from sklearn.ensemble import RandomForestRegressor # type: ignore
# from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor # type: ignore
# from utils import calculate_classes, join_target, calculate_metrics, calculate_alumnos
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from utils_model import parse_args, get_dev_version
from utils_metrics import calculate_classes , calculate_metrics
import mlflow


class TrainInicial:
    def __init__(self, 
                 input_feats_datastore:str,
                 input_target_datastore:str,
                 output_predict_datastore:str,
                 feats_version:str,
                 target_version:str
                 ):

        self.input_feats_datastore = Path(input_feats_datastore)
        self.input_target_datastore = Path(input_target_datastore)
        self.output_predict_datastore = Path(output_predict_datastore)
        self.tipo = 'inicial_regular'
        self.feats_version = feats_version
        self.target_version = target_version

    def get_data_test(self, periodo:int):
        data_model_test = pd.read_parquet(
            self.input_feats_datastore/self.tipo/"test"/self.feats_version/f"data_feats_{self.tipo}_{periodo}.parquet")
        return data_model_test

    def get_data_target(self, periodo:int):
        data_model_target = pd.read_parquet(
            self.input_target_datastore/self.tipo/"test"/self.target_version/f"data_target_{self.tipo}_{periodo}.parquet")
        return data_model_target

    def load_model(self, periodo:int):
        model = mlflow.catboost.load_model(
            f"models:/{self.tipo}_{periodo}@dev"
        )
        return model
    
    def get_data_predict(self, periodo:int, model):
        """ 
        Compute metrics for the forecast.
        
        Args:
            df_forecast (pd.DataFrame): DataFrame with forecast data.
        
        Returns:
            pd.DataFrame: DataFrame with forecast data after computing metrics.
        """

        # Loading evaluation data
        data_model_eval = self.get_data_test(periodo)
        
        meses = [1, 2, 3]

        base = ['PERIODO_TARGET', 'CURSO_ACTUAL', 'HORARIO_ACTUAL', 'IDX',
                    'PE', 'VAC_ACAD_ESTANDAR']
        
        # cat_features = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']
        cat_features = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']
        
        if periodo % 100 in meses:
            num_features = ['CANT_ALUMNOS_LAG_12', 'CANT_ALUMNOS_LAG_01', 
                            'CANT_ALUMNOS_LAG_02', 'CANT_ALUMNOS_LAG_03']
            
        else:
            num_features  = ['CANT_ALUMNOS_LAG_01', 'CANT_ALUMNOS_LAG_02', 
                             'CANT_ALUMNOS_LAG_03']

        target = 'CANT_ALUMNOS'
        predict = f"{target}_PREDICT"

        x = num_features + cat_features
        
        data_model_eval = data_model_eval[base + num_features + cat_features].copy()
        
        X_eval = data_model_eval[x].copy()

        preds = model.predict(X_eval)

        preds = np.where(preds < 0, 0, preds)
        preds = preds // 1 + np.where(preds %1 >= 0.4,1, 0)
        
        data_model_eval[predict] = preds

        data_model_eval = calculate_classes(data_model_eval)

        return data_model_eval
            
    def upload_data_predict(self, model_periodo:int, periodo:int, df_model_predict:pd.DataFrame):
        name = f"{self.tipo}_{model_periodo}"
        model_version = get_dev_version(name)
        print(f"Model Version for {name}: {model_version}")
        
        path_model_version = self.output_predict_datastore/self.tipo/"test"/f"v{model_version}"
        path_champion = self.output_predict_datastore/self.tipo/"test"/"champion"
        path_dev = self.output_predict_datastore/self.tipo/"test"/"dev"
        
        path_champion.mkdir(parents=True, exist_ok=True)
        path_dev.mkdir(parents=True, exist_ok=True)
        path_model_version.mkdir(parents=True, exist_ok=True)
        
        df_model_predict.to_parquet(
            path_model_version/f"data_predict_{self.tipo}_{periodo}.parquet")
        
        df_model_predict.to_parquet(
            path_dev/f"data_predict_{self.tipo}_{periodo}.parquet")

        if not (path_champion/f"data_predict_{self.tipo}_{periodo}.parquet").exists():
            df_model_predict.to_parquet(
                path_champion/f"data_predict_{self.tipo}_{periodo}.parquet")
        
        # if not (path_model_version/f"data_predict_{self.tipo}_{periodo}.parquet").exists():
        #     df_model_predict.to_parquet(
        #         path_model_version/f"data_predict_{self.tipo}_{periodo}.parquet"
        #     )
        # else:
        #     print(f"Already exists the predict for {self.tipo}_{periodo}")
        
   
   
    
    def logging_metrics(self, periodo:int, model, df_model_predict:pd.DataFrame):
        
        data_target_eval = self.get_data_target(periodo)
        
        on_cols = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']
        
        metrics = calculate_metrics(df_model_predict, data_target_eval, on_cols)
        # print(metrics)

        with mlflow.start_run():
            mlflow.catboost.log_model(model, name=self.tipo)
            mlflow.log_metrics(metrics, step=periodo)
            mlflow.log_params(model.get_params())


def main(args):
    
    input_feats_datastore = args.input_feats_datastore
    input_target_datastore = args.input_target_datastore
    output_predict_datastore = args.output_predict_datastore
    experiment_name = args.experiment_name
    feats_version = args.feats_version
    target_version = args.target_version
    model_periodo = args.model_periodo
    periodo = args.periodo

    # python src/predict/inicial_regular.py --input_feats_datastore $input_feats_datastore --input_target_datastore $input_target_datastore --output_predict_datastore $output_predict_datastore --experiment_name $experiment_name --model_periodo $model_periodo --periodo $periodo
    
    # listening to port 5000
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    
    train_inicial = TrainInicial(
        input_feats_datastore,
        input_target_datastore,
        output_predict_datastore,
        feats_version,
        target_version
        )
    
    model = train_inicial.load_model(model_periodo)
    df_model_predict = train_inicial.get_data_predict(periodo, model)
    train_inicial.upload_data_predict(model_periodo, periodo, df_model_predict)
    # train_inicial.logging_metrics(periodo, model, df_model_predict)
    

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