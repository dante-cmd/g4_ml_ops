import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestRegressor # type: ignore
# from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor # type: ignore
from catboost import CatBoostRegressor
#  from utils import calculate_classes, calculate_metrics, join_target, filter_by_hora_atencion
from pathlib import Path
from utils_model import parse_args
from utils_metrics import calculate_classes , calculate_metrics
import mlflow
from mlflow.client import MlflowClient


class TrainInicial:
    def __init__(self, 
                 input_feats_datastore:str,
                 feats_version:str,
                 client: MlflowClient
                 ):
        
        self.input_feats_datastore = Path(input_feats_datastore)
        self.feats_version = feats_version
        self.client = client
        self.tipo = 'inicial_estacional'
        
    def apply_filter(self, df_train:pd.DataFrame):
        return df_train
    
    def get_data_train(self, periodo:int):
        data_model_train = pd.read_parquet(
            self.input_feats_datastore/self.tipo/"train"/self.feats_version/f"data_feats_{self.tipo}_{periodo}.parquet")
        return data_model_train  

    def train_model(self, periodo:int):
        
        data_model_train = self.get_data_train(periodo)
        data_model_train = self.apply_filter(data_model_train)

        # granular = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 
        #             'HORARIO_ACTUAL', 'IDX', 'PE', 'VAC_ACAD_ESTANDAR']
        
        cat_features = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']
        
        if periodo % 100 in [1]:
            num_features = ['CANT_ALUMNOS_LAG_12']
        else:
            num_features = ['CANT_ALUMNOS_LAG_01']
        
        target = 'CANT_ALUMNOS'
        y = target
        x = cat_features + num_features
        X_train = data_model_train[x].copy()
        y_train = data_model_train[y].copy()
        
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            verbose=False, # Set to True to see training progress,
            min_data_in_leaf=5,
        )

        # X_train, y_train = data_train_01[x].copy(), data_train[y].copy()
        
        model.fit(X_train, y_train, cat_features=cat_features)
        return model
    
    def register_model(self, model, periodo:int):
        with mlflow.start_run():
            model_name = self.tipo + "_" + str(periodo)
            model_info = mlflow.catboost.log_model(
                cb_model=model,
                name=self.tipo,
                registered_model_name=model_name
            )
            versions = self.client.get_registered_model(model_name)
            if 'champion' not in versions.aliases.keys():
                self.client.set_registered_model_alias(
                    model_name, "champion", version=model_info.registered_model_version)
            
            # Assign '@champion' to Version 1
            # client.set_registered_model_alias(self.tipo, "champion", version="1")
            # Assign '@dev' to Version 1
            self.client.set_registered_model_alias(
                model_name, "dev", version=model_info.registered_model_version)


def main(args):
    
    input_feats_datastore = args.input_feats_datastore
    feats_version = args.feats_version
    experiment_name = args.experiment_name
    model_periodo = args.model_periodo
    
    # listening to port 5000
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    
    mlflow.set_experiment(experiment_name)
    # 1. Initialize client
    
    
    train_inicial = TrainInicial(
        input_feats_datastore,
        feats_version,
        client)
    
    model = train_inicial.train_model(model_periodo)
    train_inicial.register_model(model, model_periodo)
    

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