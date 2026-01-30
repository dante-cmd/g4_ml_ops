import pandas as pd
from pathlib import Path
from utils_evaluation import parse_args, calculate_metrics, join_target, get_dev_version
# from catboost import CatBoostRegressor
# import mlflow


class TrainInicial:
    def __init__(self, 
                 input_predict_datastore:str,
                 input_target_datastore:str,
                 output_evaluation_datastore:str,
                 target_version:str,
                 model_version:str
                 ):

        self.input_predict_datastore = Path(input_predict_datastore)
        self.input_target_datastore = Path(input_target_datastore)
        self.output_evaluation_datastore = Path(output_evaluation_datastore)
        self.target_version = target_version
        self.model_version= model_version
        self.tipo = 'inicial_regular'

    def get_data_predict(self, model_periodo:int, periodo:int):
        
        data_model_predict = pd.read_parquet(
            self.input_predict_datastore/'test'/self.model_version/f"data_predict_{model_periodo}_{self.tipo}_{periodo}.parquet")
        
        return data_model_predict

    def get_data_target(self, periodo:int):
        data_model_target = pd.read_parquet(
            self.input_target_datastore/'test'/self.target_version/f"data_target_{self.tipo}_{periodo}.parquet")
        return data_model_target
    
    def get_data_evaluation(self, model_periodo:int, periodo:int):
        on_cols = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']
        
        data_model_predict = self.get_data_predict(model_periodo, periodo)
        data_model_target = self.get_data_target(periodo)

        data_model_evaluation = join_target(data_model_predict, data_model_target, on_cols)

        return data_model_evaluation
    
    def upload_data_evaluation(self, model_periodo:int, model_version:str, periodo:int, df_model_evaluation:pd.DataFrame):

        # name = self.tipo + '_' + str(model_periodo)
        # version = get_dev_version(name)
        # model_version = f"v{version}"

        path_model = (self.output_evaluation_datastore/'test'/model_version)
        # path_champion = self.output_evaluation_datastore/"test"/"champion"
        # path_dev = self.output_evaluation_datastore/"test"/"dev"
        
        path_model.mkdir(parents=True, exist_ok=True)
        # path_champion.mkdir(parents=True, exist_ok=True)
        # path_dev.mkdir(parents=True, exist_ok=True)
        
        df_model_evaluation.to_parquet(
            path_model/f"data_evaluation_{model_periodo}_{self.tipo}_{periodo}.parquet"
        )
        
        df_model_evaluation.to_parquet(
            path_model/f"data_evaluation_dev_{model_periodo}_{self.tipo}_{periodo}.parquet"
        )
        
        if not (path_model/f"data_evaluation_champion_{model_periodo}_{self.tipo}_{periodo}.parquet").exists():
            df_model_evaluation.to_parquet(
                path_model/f"data_evaluation_champion_{model_periodo}_{self.tipo}_{periodo}.parquet"
            )


def main(args):
    
    input_predict_datastore = args.input_predict_datastore
    input_target_datastore = args.input_target_datastore
    output_evaluation_datastore = args.output_evaluation_datastore
    target_version = args.target_version
    model_periodo = args.model_periodo
    model_version = args.model_version
    periodo = args.periodo
    
    train_inicial = TrainInicial(
        input_predict_datastore,
        input_target_datastore,
        output_evaluation_datastore,
        target_version,
        model_version)

    df_model_evaluation = train_inicial.get_data_evaluation(model_periodo,periodo)
    train_inicial.upload_data_evaluation(model_periodo, model_version,periodo, df_model_evaluation)


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