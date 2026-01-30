import pandas as pd

# from sklearn.ensemble import RandomForestRegressor # type: ignore
# from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor # type: ignore
# from utils import calculate_classes, join_target, calculate_metrics, calculate_alumnos
import numpy as np
from pathlib import Path
import argparse

from catboost import CatBoostRegressor
# from utils_model import get_dev_version
from utils_metrics import calculate_classes, calculate_metrics
# import mlflow
# import mlflow.sklearn

# import azureml.mlflow
# from mlflow.tracking import MlflowClient


class TrainInicial:
    def __init__(
        self,
        input_model_datastore: str,
        input_feats_datastore: str,
        input_target_datastore: str,
        output_predict_datastore: str,
        feats_version: str,
        target_version: str,
        # model_periodo: int,
        model_version: str,
    ):

        self.input_model_datastore = input_model_datastore
        self.input_feats_inicial_datastore = Path(input_feats_inicial_datastore)
        self.input_target_inicial_datastore = Path(input_target_inicial_datastore)
        self.output_predict_inicial_datastore = Path(output_predict_inicial_datastore)
        self.tipo = "inicial_regular"
        self.feats_version = feats_version
        self.target_version = target_version
        # self.model_periodo = model_periodo
        self.model_version = model_version

    def get_data_test(self, periodo: int):
        data_model_test = pd.read_parquet(
            self.input_feats_inicial_datastore
            / "test"
            / self.feats_version
            / f"data_feats_{self.tipo}_{periodo}.parquet"
        )
        return data_model_test

    def get_data_target(self, periodo: int):
        data_model_target = pd.read_parquet(
            self.input_target_inicial_datastore
            / "test"
            / self.target_version
            / f"data_target_{self.tipo}_{periodo}.parquet"
        )
        return data_model_target

    def load_model(self, model_periodo:int):
        print(f"Cargando modelo desde: {self.input_model_datastore}")
        model = CatBoostRegressor()
        model.load_model(self.input_model_datastore/self.model_version/f"{self.tipo}_{model_periodo}.cbm")
        # model = mlflow.sklearn.load_model(
        #     self.input_model_datastore
        #     # f"models:/{self.tipo}_{periodo}@dev"
        # )
        return model

    def get_data_predict(self, periodo: int, model):
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
        # data_model_eval = pd.DataFrame(
        #     data=np.random.randint(0, 2, size=(100, 3)),
        #     columns=["PERIODO_TARGET", "CURSO_ACTUAL", "HORARIO_ACTUAL"],
        # )

        return data_model_eval

    def upload_data_predict(
        self, model_version: str, periodo: int, df_model_predict: pd.DataFrame
    ):
        # name = f"{self.tipo}_{model_periodo}"
        # model_version = get_dev_version(name, self.client)
        # print(f"Model Version for {name}: {model_version}")
        print(self.output_predict_inicial_datastore)
        print(model_version)
        print(periodo)
        print(df_model_predict)

        path_model_version = (
            self.output_predict_inicial_datastore / model_version / "test"
        )
        path_champion = self.output_predict_inicial_datastore / model_version / "champion"
        path_dev = self.output_predict_inicial_datastore / model_version / "dev"

        path_champion.mkdir(parents=True, exist_ok=True)
        path_dev.mkdir(parents=True, exist_ok=True)
        path_model_version.mkdir(parents=True, exist_ok=True)

        df_model_predict.to_parquet(
            path_model_version / f"data_predict_{self.tipo}_{periodo}.parquet"
        )

        df_model_predict.to_parquet(
            path_dev / f"data_predict_{self.tipo}_{periodo}.parquet"
        )

        if not (path_champion / f"data_predict_{self.tipo}_{periodo}.parquet").exists():
            df_model_predict.to_parquet(
                path_champion / f"data_predict_{self.tipo}_{periodo}.parquet"
            )

        # if not (path_model_version/f"data_predict_{self.tipo}_{periodo}.parquet").exists():
        #     df_model_predict.to_parquet(
        #         path_model_version/f"data_predict_{self.tipo}_{periodo}.parquet"
        #     )
        # else:
        #     print(f"Already exists the predict for {self.tipo}_{periodo}")

    # def logging_metrics(self, periodo:int, model, df_model_predict:pd.DataFrame):

    #     data_target_eval = self.get_data_target(periodo)

    #     on_cols = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']

    #     metrics = calculate_metrics(df_model_predict, data_target_eval, on_cols)
    #     # print(metrics)

    #     with mlflow.start_run():
    #         mlflow.sklearn.log_model(model, name=self.tipo)
    #         mlflow.log_metrics(metrics, step=periodo)
    # mlflow.log_params(model.get_params())


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_model_datastore",
        dest="input_model_datastore",
        type=str,
        help="Ruta del modelo (artifact)",
    )
    parser.add_argument(
        "--input_feats_datastore",
        dest="input_feats_datastore",
        type=str,
    )
    parser.add_argument(
        "--input_target_datastore",
        dest="input_target_datastore",
        type=str,
    )
    parser.add_argument(
        "--output_predict_datastore",
        dest="output_predict_datastore",
        type=str,
    )
    parser.add_argument("--feats_version", dest="feats_version", type=str)
    parser.add_argument("--target_version", dest="target_version", type=str)
    parser.add_argument("--periodo", dest="periodo", type=int)
    parser.add_argument("--model_periodo", dest="model_periodo", type=int)
    parser.add_argument("--model_version", dest="model_version", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args):

    input_model_datastore = args.input_model_datastore
    input_feats_datastore = args.input_feats_datastore
    input_target_datastore = args.input_target_datastore
    output_predict_datastore = args.output_predict_datastore
    # experiment_name = args.experiment_name
    feats_version = args.feats_version
    target_version = args.target_version
    model_periodo = args.model_periodo
    model_version = args.model_version
    periodo = args.periodo

    # credential = DefaultAzureCredential()

    # Conectarse al Workspace actual (se infiere del entorno de ejecución)
    # Nota: En producción, pasas subscription/group/workspace como args o variables de entorno
    # ml_client = MLClient.from_config(credential=credential)

    # Crear el objeto Modelo
    # model = Model(
    #     path=args.model_input_path,
    #     name=args.model_name,
    #     description="Registrado automáticamente via Azure ML Pipeline",
    #     type=AssetTypes.MLFLOW_MODEL
    # )

    # Registrar en Azure ML
    # registered_model = ml_client.models.create_or_update(model)
    # print(f"Modelo registrado versión: {registered_model.version}")
    print(f"Cargando modelo desde: {args.input_model_datastore}")

    # python src/predict/inicial_regular.py --input_feats_datastore $input_feats_datastore --input_target_datastore $input_target_datastore --output_predict_datastore $output_predict_datastore --experiment_name $experiment_name --model_periodo $model_periodo --periodo $periodo

    # listening to port 5000
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # client = MlflowClient("http://127.0.0.1:5000")
    # client = MlflowClient()

    # mlflow.set_experiment(experiment_name)

    train_inicial = TrainInicial(
        input_model_datastore,
        input_feats_datastore,
        input_target_datastore,
        output_predict_datastore,
        feats_version,
        target_version,
        model_version,
    )

    model = train_inicial.load_model(model_periodo)
    df_model_predict = train_inicial.get_data_predict(periodo, model)
    train_inicial.upload_data_predict(model_version, periodo, df_model_predict)
    # train_inicial.logging_metrics(periodo, model, df_model_predict)


if __name__ == "__main__":
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
