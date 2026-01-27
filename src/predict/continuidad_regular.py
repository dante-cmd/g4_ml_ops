import pandas as pd
# from sklearn.ensemble import RandomForestRegressor # type: ignore
# from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor # type: ignore
# from utils import calculate_classes, join_target, calculate_metrics, calculate_alumnos
import numpy as np
from catboost import CatBoostRegressor, Pool
from pathlib import Path
from utils_model import parse_args, get_dev_version
from utils_metrics import calculate_classes , calculate_metrics, fac_to_cant
import mlflow


class TrainContinuidad:
    def __init__(self,
                 input_feats_datastore:str,
                 input_target_datastore:str,
                 output_predict_datastore:str,
                 feats_version:str,
                 target_version:str,
                 ):
        
        self.input_feats_datastore = Path(input_feats_datastore)
        self.input_target_datastore = Path(input_target_datastore)
        self.output_predict_datastore = Path(output_predict_datastore)
        self.tipo  = 'continuidad_regular'
        self.feats_version = feats_version
        self.target_version = target_version

    # def apply_filter(self, df_train:pd.DataFrame):
    #     return df_train
    
    # def get_data_train(self, periodo:int):
    #     data_model_train = pd.read_parquet(
    #         self.input_feats_train_datastore/f"data_feats_train_{self.tipo}_{periodo}.parquet")
    #     return data_model_train  
    
    def get_data_test(self, periodo:int):
        data_model_test = pd.read_parquet(
            self.input_feats_datastore/self.tipo/"test"/self.feats_version/f"data_feats_{self.tipo}_{periodo}.parquet")
        return data_model_test

    def get_data_target(self, periodo:int):
        data_model_target = pd.read_parquet(
            self.input_target_datastore/self.tipo/"test"/self.target_version/f"data_target_{self.tipo}_{periodo}.parquet")
        return data_model_target

    # def training_model(self, periodo:int):

    #     data_model_train = self.get_data_train(periodo)
    #     data_model_train = self.apply_filter(data_model_train)
        
    #     meses = [1, 2, 3]
    #     # granular = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'IDX', 'PE', 'VAC_ACAD_ESTANDAR']

    #     cat_features = ['LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']

    #     if periodo % 100 in meses:
    #         # cat_features = ['LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']
    #         # target = "PCT_ALUMNOS"
    #         # x = num_features + cat_features
    #         # y = target
    #         if periodo%100 in [1, 3]:
    #             num_features = [
    #                 'CANT_ALUMNOS_LAG_12', 
    #                 'CANT_ALUMNOS_ANTERIOR',
    #                 # 'DIFF_ALUMNOS_SPLY', 
    #                 'PCT_ALUMNOS_SPLY']
    #             target = "PCT_ALUMNOS"
    #             x = num_features + cat_features
    #             y = target
    #         else:
    #             num_features = [
    #                 # 'CANT_ALUMNOS_LAG_12', 
    #                 'CANT_ALUMNOS_ANTERIOR', 
    #                 # 'DIFF_ALUMNOS_SPLY', 
    #                 # 'PCT_ALUMNOS_SPLY'
    #                 ]
    #             target = "CANT_ALUMNOS"
    #             x = num_features + cat_features
    #             y = target
            
    #         data_model_train = data_model_train[
    #             # (X_train['PERIODO_TARGET'] < periodo)
    #             (data_model_train['PERIODO_TARGET'] % 100).isin([periodo % 100])
    #         ].copy()
            
    #         X_train = data_model_train[x].copy()
    #         y_train = data_model_train[y].copy()

    #     else:
    #         num_features = ['CANT_ALUMNOS_ANTERIOR']
    #         target = "CANT_ALUMNOS"
    #         x = num_features + cat_features
    #         y = target
    #         data_model_train = data_model_train[
    #             # (X_train['PERIODO_TARGET'] < periodo)
    #             ~(data_model_train['PERIODO_TARGET'] % 100).isin(meses)
    #         ].copy()
    #         X_train = data_model_train[x].copy()
    #         y_train = data_model_train[y].copy()
        
    #      # for target in targets:
    #     # x = numerical_columns + self.dummy_columns
    #     # 4. Initialize and Train CatBoostRegressor
    #     model = CatBoostRegressor(
    #         iterations=500,
    #         learning_rate=0.1,
    #         depth=6,
    #         loss_function='RMSE',
    #         verbose=False, # Set to True to see training progress,
    #         min_data_in_leaf=5,
    #     )

    #     # X_train, y_train = data_train_01[x].copy(), data_train[y].copy()
        
    #     model.fit(X_train, y_train, cat_features=cat_features)
    #     return model

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
        
        # Load logging
        # log_metrics = self.get_logging()

        # Load model
        # model = CatBoostRegressor()
        # model.load_model(self.input_model_datastore/f"{self.tipo}_{self.periodo_model}.cbm")

        # read data model
        data_model_eval = self.get_data_test(periodo)
        # data_model_eval = pd.read_parquet(
        #     self.input_feats_test_datastore/f'data_feats_test_{self.tipo}_{periodo}.parquet')
        # target = 'CANT_ALUMNOS'
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
    
    def logging_metrics(self, periodo:int, model, df_model_predict:pd.DataFrame):
        
        data_target_eval = self.get_data_target(periodo)
        
        on_cols = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL']
        
        metrics = calculate_metrics(df_model_predict, data_target_eval, on_cols)
        # print(metrics)

        with mlflow.start_run():
            mlflow.catboost.log_model(model, name=self.tipo)
            mlflow.log_metrics(metrics, step=periodo)
            mlflow.log_params(model.get_params())

        # model.save_model(self.output_model_datastore /f"{self.tipo}_{periodo}.cbm")

        # medidas = ['ALUMNOS']
        # predict_columns = ["CANT" + '_' + medida + '_' + 'PREDICT' for medida in medidas]

        # feature_numeric_columns = list(numerical_columns.values())
        
        # collection = []
        # for feat_num_col in feature_numeric_columns:
        #     collection.extend(feat_num_col)
        # for medida in medidas:
        #     target = tipo + '_' + medida
        #     x = numerical_columns[medida] + self.dummy_columns
        #     hgr = HistGradientBoostingRegressor(random_state=42)
        #     hgr.fit(data_train[x], data_train[target])
        #     predict = hgr.predict(data_eval[x])
        #     if tipo == "PCT":
        #         data_eval[f'CANT_{medida}_PREDICT'] = data_eval [f'CANT_{medida}_ANTERIOR'] * (1 + predict)
        #     elif tipo == "DIFF":
        #         data_eval[f'CANT_{medida}_PREDICT'] = data_eval [f'CANT_{medida}_ANTERIOR'] + predict
        #     else:
        #         data_eval[f'CANT_{medida}_PREDICT'] = predict
            
        #     data_eval[f'CANT_{medida}_PREDICT'] = np.where(
        #         data_eval[f'CANT_{medida}_PREDICT'] < 0, 
        #         0, 
        #         data_eval[f'CANT_{medida}_PREDICT']//1 + np.where(data_eval[f'CANT_{medida}_PREDICT']%1 >= 0.4, 1, 0))

        # data_eval_01 = data_eval[granular + collection + predict_columns].copy()
        # # print(self.__str__())
        # return data_eval_01
       

class TrainContinuidadToHorario:
    def __init__(self,
                 input_feats_datastore:str,
                 input_target_datastore:str,
                 output_predict_datastore:str,
                 feats_version:str,
                 target_version:str,
                 df_predict_continuidad:pd.DataFrame
                 ):
        
        self.input_feats_datastore = Path(input_feats_datastore)
        self.input_target_datastore = Path(input_target_datastore)
        self.output_predict_datastore = Path(output_predict_datastore)
        self.df_predict_continuidad = df_predict_continuidad
        self.tipo = 'continuidad_regular_horario'
        self.feats_version = feats_version
        self.target_version = target_version
    
    # def apply_filter(self, df_train:pd.DataFrame):
    #     return df_train
    
    # def get_data_train(self, periodo:int):
    #     data_model_train = pd.read_parquet(
    #         self.input_feats_train_datastore/f"data_feats_train_{self.tipo}_{periodo}.parquet")
    #     return data_model_train  

    def get_data_test(self, periodo:int):
        data_model_test = pd.read_parquet(
            self.input_feats_datastore/self.tipo/"test"/self.feats_version/f"data_feats_{self.tipo}_{periodo}.parquet")
        return data_model_test

    def get_data_target(self, periodo:int):
        data_model_target = pd.read_parquet(
            self.input_target_datastore/self.tipo/"test"/self.target_version/f"data_target_{self.tipo}_{periodo}.parquet")
        return data_model_target
    
    # def train_model(self, periodo:int):
    #     # self.logging.info(f"training {self.name_model}".center(50, '-'))
    #     data_model_train = self.get_data_train(periodo)
    #     data_model_train = self.apply_filter(data_model_train)
    #     meses = [1, 2, 3]
    #     # granular = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL','HORARIO_ACTUAL',
    #     #             'IDX', 'PE', 'VAC_ACAD_ESTANDAR']
        
    #     cat_features = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']
    #     target = 'FAC_ALUMNOS'

    #     if periodo % 100 in meses:
    #         num_features = ['FAC_ALUMNOS_LAG_12', 'FAC_ALUMNOS_ANTERIOR']
    #         x = num_features + cat_features
    #         y = target
    #         # numerical_columns = {'FAC_CLASES': ['FAC_CLASES_LAG_12', 'FAC_CLASES_ANTERIOR'],
    #         #                      'FAC_ALUMNOS': ['FAC_ALUMNOS_LAG_12', 'FAC_ALUMNOS_ANTERIOR']}
    #         data_model_train = data_model_train[
    #             (data_model_train['PERIODO_TARGET'] % 100).isin([periodo % 100])
    #             # (self.df_model['PERIODO_TARGET'] < periodo)
    #             # & 
    #         ].copy()
            
    #         X_train = data_model_train[x].copy()
    #         y_train = data_model_train[y].copy()

    #     else:
    #         num_features = ['FAC_ALUMNOS_ANTERIOR']
    #         x = num_features + cat_features
    #         y = target
            
    #         data_model_train = data_model_train[
    #             ~(data_model_train['PERIODO_TARGET'] % 100).isin(meses)
    #         ].copy()

    #         X_train = data_model_train[x].copy()
    #         y_train = data_model_train[y].copy()
            

    #         # data_eval = data_model_train[data_model_train['PERIODO_TARGET'] == periodo].copy()

    #     model = CatBoostRegressor(
    #         iterations=500,
    #         learning_rate=0.1,
    #         depth=6,
    #         loss_function='RMSE',
    #         verbose=False, # Set to True to see training progress,
    #         min_data_in_leaf=5,
    #     )

    #     # X_train, y_train = data_train_01[x].copy(), data_train[y].copy()
        
    #     model.fit(X_train, y_train, cat_features=cat_features)
        
    #     return model

    def get_data_predict(self, periodo:int, model):
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
        data_model_eval = fac_to_cant(data_model_eval, self.df_predict_continuidad)

        data_model_eval = calculate_classes(data_model_eval)

        return data_model_eval

                
    def load_model(self, periodo:int):
        model = mlflow.catboost.load_model(
            f"models:/{self.tipo}_{periodo}@dev"
        )
        return model
    
    
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
    # input_feats_train_datastore = args.input_feats_train_datastore
    input_feats_datastore = args.input_feats_datastore
    input_target_datastore = args.input_target_datastore
    output_predict_datastore = args.output_predict_datastore
    experiment_name = args.experiment_name
    feats_version = args.feats_version
    target_version = args.target_version
    model_periodo = args.model_periodo
    periodo = args.periodo
    
    # listening to port 5000
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    
    train_continuidad = TrainContinuidad(
        input_feats_datastore,
        input_target_datastore,
        output_predict_datastore,
        feats_version,
        target_version
        )
    
    model = train_continuidad.load_model(model_periodo)
    df_model_predict = train_continuidad.get_data_predict(periodo, model)
    train_continuidad.upload_data_predict(model_periodo, periodo, df_model_predict)
    # train_continuidad.logging_metrics(periodo, model, df_model_predict)

    train_continuidad_horario = TrainContinuidadToHorario(
        input_feats_datastore,
        input_target_datastore,
        output_predict_datastore,
        feats_version,
        target_version,
        df_model_predict)
    
    model_horario = train_continuidad_horario.load_model(model_periodo)
    df_model_predict_horario = train_continuidad_horario.get_data_predict(periodo, model_horario)
    train_continuidad_horario.upload_data_predict(model_periodo, periodo, df_model_predict_horario)
    # train_continuidad_horario.logging_metrics(periodo, model_horario, df_model_predict_horario) 
    
    # input_feats_train_datastore = args.input_feats_train_datastore
    # output_model_datastore = args.output_model_datastore
    # periodo = args.periodo
    
    # train_continuidad = TrainContinuidad(
    #     input_feats_train_datastore, 
    #     output_model_datastore)
    # model = train_continuidad.training(periodo)

    # train_continuidad_horario = TrainContinuidadToHorario(
    #     input_feats_train_datastore, output_model_datastore)
    # model_horario = train_continuidad_horario.training(periodo)
    

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