import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from utils_model import parse_args, get_dev_version
from utils_metrics import calculate_classes , calculate_metrics
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
        self.tipo  = 'continuidad_estacional'
        self.feats_version = feats_version
        self.target_version = target_version

    # def apply_filter(self, df_train:pd.DataFrame):
    #     return df_train.copy()

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
    
    # def training_model(self, periodo):
        
    #     # granular = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'CURSO_ANTERIOR',
    #     #             'HORARIO_ACTUAL', 'IDX', 'PE', 'VAC_ACAD_ESTANDAR']

    #     data_model_train = self.get_data_train(periodo)
        
    #     data_model_train = self.apply_filter(data_model_train)

    #     cat_features = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']

    #     num_features = ['CANT_ALUMNOS_ANTERIOR']

    #     target = 'CANT_ALUMNOS'

    #     x = num_features + cat_features
    #     y = target

    #     # for target in targets:
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

    #     X_train, y_train = data_model_train[x].copy(), data_model_train[y].copy()
        
    #     model.fit(X_train, y_train, cat_features=cat_features)
        
    #     return model

    #     # model.save_model(self.output_model_datastore / f"{self.tipo}_{periodo}.cbm")

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
        # log_metrics = self.get_logging()

        base = ['PERIODO_TARGET', 'CURSO_ACTUAL', 'CURSO_ANTERIOR',
                    'HORARIO_ACTUAL', 'IDX', 'PE', 'VAC_ACAD_ESTANDAR']
        
        cat_features = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']

        num_features = ['CANT_ALUMNOS_ANTERIOR']

        target = 'CANT_ALUMNOS'
        predict = f"{target}_PREDICT"

        x = num_features + cat_features
        
        # model = CatBoostRegressor()

        # model.load_model(self.input_model_datastore/f"{self.tipo}_{self.periodo_model}.cbm")
        
        # data_model_eval = pd.read_parquet(
        #     self.input_feats_test_datastore/f'data_feats_test_{self.tipo}_{periodo}.parquet')
        
        data_model_eval = self.get_data_test(periodo)
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
        print(metrics)

        with mlflow.start_run():
            mlflow.catboost.log_model(model, name=self.tipo)
            mlflow.log_metrics(metrics, step=periodo)
            mlflow.log_params(model.get_params())


def main(args):
    input_feats_datastore = args.input_feats_datastore
    input_target_datastore = args.input_target_datastore
    output_predict_datastore = args.output_predict_datastore
    feats_version = args.feats_version
    target_version = args.target_version
    experiment_name = args.experiment_name
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
        target_version,
        )
    
    model = train_continuidad.load_model(model_periodo)
    df_model_predict = train_continuidad.get_data_predict(periodo, model)
    train_continuidad.upload_data_predict(model_periodo, periodo, df_model_predict)
    # train_continuidad.logging_metrics(periodo, model, df_model_predict)
    
    # input_feats_train_datastore = args.input_feats_train_datastore
    # output_model_datastore = args.output_model_datastore
    # periodo = args.periodo
    
    # train_continuidad = TrainContinuidad(input_feats_train_datastore, output_model_datastore)
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