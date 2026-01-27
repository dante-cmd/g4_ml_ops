import pandas as pd
# from sklearn.ensemble import RandomForestRegressor # type: ignore
# from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor # type: ignore
# from utils import calculate_classes, join_target, calculate_metrics, calculate_alumnos
import numpy as np
from catboost import CatBoostRegressor, Pool
from pathlib import Path
from utils_model import parse_args
from utils_metrics import calculate_classes , calculate_metrics, fac_to_cant
import mlflow
from mlflow.tracking import MlflowClient


class TrainContinuidad:
    def __init__(self,
                 input_feats_datastore:str,
                 feats_version:str,
                 client:MlflowClient,
                 ):
        
        self.input_feats_datastore = Path(input_feats_datastore)
        self.feats_version = feats_version
        self.client = client
        self.tipo  = 'continuidad_regular'

    def apply_filter(self, df_train:pd.DataFrame):
        return df_train
        # MlflowClient, MLflowClient
    
    def get_data_train(self, periodo:int):
        data_model_train = pd.read_parquet(
            self.input_feats_datastore/f"{self.tipo}/train/{self.feats_version}/data_feats_{self.tipo}_{periodo}.parquet")
        return data_model_train  
    
    def train_model(self, periodo:int):

        data_model_train = self.get_data_train(periodo)
        data_model_train = self.apply_filter(data_model_train)
        
        meses = [1, 2, 3]
        # granular = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'IDX', 'PE', 'VAC_ACAD_ESTANDAR']

        cat_features = ['LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']

        if periodo % 100 in meses:
            # cat_features = ['LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']
            # target = "PCT_ALUMNOS"
            # x = num_features + cat_features
            # y = target
            if periodo%100 in [1, 3]:
                num_features = [
                    'CANT_ALUMNOS_LAG_12', 
                    'CANT_ALUMNOS_ANTERIOR',
                    # 'DIFF_ALUMNOS_SPLY', 
                    'PCT_ALUMNOS_SPLY']
                target = "PCT_ALUMNOS"
                x = num_features + cat_features
                y = target
            else:
                num_features = [
                    # 'CANT_ALUMNOS_LAG_12', 
                    'CANT_ALUMNOS_ANTERIOR', 
                    # 'DIFF_ALUMNOS_SPLY', 
                    # 'PCT_ALUMNOS_SPLY'
                    ]
                target = "CANT_ALUMNOS"
                x = num_features + cat_features
                y = target
            
            data_model_train = data_model_train[
                # (X_train['PERIODO_TARGET'] < periodo)
                (data_model_train['PERIODO_TARGET'] % 100).isin([periodo % 100])
            ].copy()
            
            X_train = data_model_train[x].copy()
            y_train = data_model_train[y].copy()

        else:
            num_features = ['CANT_ALUMNOS_ANTERIOR']
            target = "CANT_ALUMNOS"
            x = num_features + cat_features
            y = target
            data_model_train = data_model_train[
                # (X_train['PERIODO_TARGET'] < periodo)
                ~(data_model_train['PERIODO_TARGET'] % 100).isin(meses)
            ].copy()
            X_train = data_model_train[x].copy()
            y_train = data_model_train[y].copy()
        
         # for target in targets:
        # x = numerical_columns + self.dummy_columns
        # 4. Initialize and Train CatBoostRegressor
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

class TrainContinuidadToHorario:
    def __init__(self,
                 input_feats_datastore:str,
                 feats_version:str,
                 client:MlflowClient
                 ):
        
        self.input_feats_datastore = Path(input_feats_datastore)
        self.feats_version = feats_version
        self.client = client
        self.tipo = 'continuidad_regular_horario'
    
    def apply_filter(self, df_train:pd.DataFrame):
        return df_train
    
    def get_data_train(self, periodo:int):
        data_model_train = pd.read_parquet(
            self.input_feats_datastore/self.tipo/"train"/self.feats_version/f"data_feats_{self.tipo}_{periodo}.parquet")
        return data_model_train  
    
    def train_model(self, periodo:int):
        # self.logging.info(f"training {self.name_model}".center(50, '-'))
        data_model_train = self.get_data_train(periodo)
        data_model_train = self.apply_filter(data_model_train)
        meses = [1, 2, 3]
        # granular = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL','HORARIO_ACTUAL',
        #             'IDX', 'PE', 'VAC_ACAD_ESTANDAR']
        
        cat_features = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']
        target = 'FAC_ALUMNOS'

        if periodo % 100 in meses:
            num_features = ['FAC_ALUMNOS_LAG_12', 'FAC_ALUMNOS_ANTERIOR']
            x = num_features + cat_features
            y = target
            # numerical_columns = {'FAC_CLASES': ['FAC_CLASES_LAG_12', 'FAC_CLASES_ANTERIOR'],
            #                      'FAC_ALUMNOS': ['FAC_ALUMNOS_LAG_12', 'FAC_ALUMNOS_ANTERIOR']}
            data_model_train = data_model_train[
                (data_model_train['PERIODO_TARGET'] % 100).isin([periodo % 100])
                # (self.df_model['PERIODO_TARGET'] < periodo)
                # & 
            ].copy()
            
            X_train = data_model_train[x].copy()
            y_train = data_model_train[y].copy()

        else:
            num_features = ['FAC_ALUMNOS_ANTERIOR']
            x = num_features + cat_features
            y = target
            
            data_model_train = data_model_train[
                ~(data_model_train['PERIODO_TARGET'] % 100).isin(meses)
            ].copy()

            X_train = data_model_train[x].copy()
            y_train = data_model_train[y].copy()
            

            # data_eval = data_model_train[data_model_train['PERIODO_TARGET'] == periodo].copy()

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
    
    train_continuidad = TrainContinuidad(
        input_feats_datastore,
        feats_version,
        client)

        # input_feats_test_datastore,
        # input_target_test_datastore)
    
    model = train_continuidad.train_model(model_periodo)
    train_continuidad.register_model(model, model_periodo)

    # df_model_predict = train_continuidad.get_data_predict(periodo, model)
    # train_continuidad.logging_metrics(periodo, model, df_model_predict)

    train_continuidad_horario = TrainContinuidadToHorario(
        input_feats_datastore,
        feats_version,
        client
        )
    
    model_horario = train_continuidad_horario.train_model(model_periodo)
    train_continuidad_horario.register_model(model_horario, model_periodo)
    

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