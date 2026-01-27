import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestRegressor # type: ignore
# from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor # type: ignore
from catboost import CatBoostRegressor
#  from utils import calculate_classes, calculate_metrics, join_target, filter_by_hora_atencion
from pathlib import Path
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
        self.tipo = 'inicial_estacional'
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
    
    
    def load_model(self, periodo:int):
        model = mlflow.catboost.load_model(
            f"models:/{self.tipo}_{periodo}@dev"
        )
        return model
    
    # def training_model(self, periodo:int):
        
    #     data_model_train = self.get_data_train(periodo)
    #     data_model_train = self.apply_filter(data_model_train)

    #     # granular = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 
    #     #             'HORARIO_ACTUAL', 'IDX', 'PE', 'VAC_ACAD_ESTANDAR']
        
    #     cat_features = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']
        
    #     if periodo % 100 in [1]:
    #         num_features = ['CANT_ALUMNOS_LAG_12']
    #     else:
    #         num_features = ['CANT_ALUMNOS_LAG_01']
        
    #     target = 'CANT_ALUMNOS'
    #     y = target
    #     x = cat_features + num_features
    #     X_train = data_model_train[x].copy()
    #     y_train = data_model_train[y].copy()
        
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
        # log_metrics = self.get_logging()

        base = ['PERIODO_TARGET', 'CURSO_ACTUAL', 
                    'HORARIO_ACTUAL', 'IDX', 'PE', 'VAC_ACAD_ESTANDAR']
        
        cat_features = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']

        if periodo % 100 in [1]:
            num_features = ['CANT_ALUMNOS_LAG_12']
        else:
            num_features = ['CANT_ALUMNOS_LAG_01']

        target = 'CANT_ALUMNOS'
        predict = f"{target}_PREDICT"

        x = cat_features + num_features
        
        # model = CatBoostRegressor()

        # model.load_model(self.input_model_datastore/f"{self.tipo}_{self.periodo_model}.cbm")
        
        data_model_eval = self.get_data_test(periodo)
        # data_model_eval = pd.read_parquet(
        #     self.input_feats_test_datastore/f'data_feats_test_{self.tipo}_{periodo}.parquet')
        
        # data_model_test.to_parquet(
        #     f"{output_feats_test_datastore}/data_feats_test_{self.tipo}_{periodo}.parquet", index=False)
        # df_real_09.to_parquet(
        #    f"{output_target_test_datastore}/data_target_test_{self.tipo}_{periodo}.parquet", index=False)
        
        data_model_eval = data_model_eval[base + num_features + cat_features].copy()
        
        X_eval = data_model_eval[x].copy()

        preds = model.predict(X_eval)

        preds = np.where(preds < 0, 0, preds)
        preds = preds // 1 + np.where(preds %1 >= 0.4,1, 0)
        
        data_model_eval[predict] = preds

        on_cols = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']

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
        
        on_cols = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']
        
        metrics = calculate_metrics(df_model_predict, data_target_eval, on_cols)
        # print(metrics)

        with mlflow.start_run():
            mlflow.catboost.log_model(model, name=self.tipo)
            mlflow.log_metrics(metrics, step=periodo)
            mlflow.log_params(model.get_params())
        # log_metrics.info(f'Mode: {mode}')
        # log_metrics.info(f' Mode: {mode} {periodo}'.center(50, '=').upper())
        # log_metrics.info(f'Tipo: {self.tipo}'.upper())

        # if mode == 'dev':
        #     data_target_eval = pd.read_parquet(
        #         self.input_target_test_datastore/f'data_target_test_{self.tipo}_{periodo}.parquet')
        #     
        #     data_model_eval_join = join_target(data_model_eval, data_target_eval, on_cols)
# 
        #     save_forecast(data_model_eval_join, self.output_forecast_datastore, self.tipo, periodo, self.periodo_model)
# 
        #     calculate_metrics(data_model_eval, data_target_eval, on_cols, log_metrics)
        # else:
        #     save_forecast(data_model_eval, self.output_forecast_datastore, self.tipo, periodo, self.periodo_model)
        #     log_metrics.info("Total Clases Predict: {:,.0f}".format(
        #         data_model_eval['CANT_CLASES_PREDICT'].sum()))
        #     log_metrics.info("Total alumnos Predict: {:,.0f}".format(
        #         data_model_eval['CANT_ALUMNOS_PREDICT'].sum()))
        # log_metrics.info(''.center(50, '='))
        # model.save_model(self.output_model_datastore/f"{self.tipo}_{periodo}.cbm")

    # def training_01(self):
    #     self.logging.info(f"training {self.name_model}".center(50, '-'))

    #     granular = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 
    #                 'HORARIO_ACTUAL', 'IDX', 'PE', 'VAC_ACAD_ESTANDAR']
    #     if self.periodo % 100 in [1]:
    #         numerical_columns = {'CANT_CLASES': ['CANT_CLASES_LAG_12'],
    #                              'CANT_ALUMNOS': ['CANT_ALUMNOS_LAG_12']}
    #     # self.periodo % 100 in [2]
    #     else:
    #         numerical_columns = {'CANT_CLASES': ['CANT_CLASES_LAG_01'],
    #                              'CANT_ALUMNOS': ['CANT_ALUMNOS_LAG_01']}
    #     targets = ['CANT_ALUMNOS']
    #     # 'CANT_CLASES',
    #     predict_columns = {target: target + '_' + 'PREDICT' for target in targets}
    #     feature_numeric_columns = list(numerical_columns.values())
        
    #     collection = []
    #     for feat_num_col in feature_numeric_columns:
    #         collection.extend(feat_num_col)

    #     data_train = self.df_model[self.df_model['PERIODO_TARGET'] < self.periodo].copy()
    #     data_eval = self.df_model[self.df_model['PERIODO_TARGET'] == self.periodo].copy()

    #     for target in targets:
    #         x = numerical_columns[target] + self.dummy_columns
    #         hgb = HistGradientBoostingRegressor()
    #         hgb.fit(data_train[x], data_train[target])
    #         predict = hgb.predict(data_eval[x])
    #         predict = np.where(predict<0, 0, predict)
    #         data_eval[predict_columns[target]] = predict//1 + np.where(predict%1 >= 0.4, 1, 0)

    #     data_eval_01 = data_eval[granular + collection + list(predict_columns.values())].copy()
    #     # print(self.__str__())
    #     return data_eval_01


def main(args):
    
    input_feats_datastore = args.input_feats_datastore
    input_target_datastore = args.input_target_datastore
    output_predict_datastore = args.output_predict_datastore
    experiment_name = args.experiment_name
    model_periodo = args.model_periodo
    periodo = args.periodo
    feats_version = args.feats_version
    target_version = args.target_version
    
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

    # input_feats_train_datastore = args.input_feats_train_datastore
    # output_model_datastore = args.output_model_datastore
    # periodo = args.periodo
    
    # train_continuidad = TrainInicial(input_feats_train_datastore, output_model_datastore)
    # model = train_continuidad.training(periodo)
    

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