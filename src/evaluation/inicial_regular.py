"""Script para evaluar el modelo de continuidad regular."""
import pandas as pd
from pathlib import Path
from utils_evaluation import parse_args, calculate_metrics, join_target, get_ahead_n_periodos, get_mapping_tipos, treshold_tipos
# from azureml.core import Run
# import mlflow


class TrainInicial:
    """Clase para evaluar el modelo de continuidad regular."""
    def __init__(self, 
                 input_predict_datastore:str,
                 input_target_datastore:str,
                 output_evaluation_datastore:str,
                 target_version:str,
                 model_version:str,
                 tipo:str
                 ):

        self.input_predict_datastore = Path(input_predict_datastore)
        self.input_target_datastore = Path(input_target_datastore)
        self.output_evaluation_datastore = Path(output_evaluation_datastore)
        self.target_version = target_version
        self.model_version= model_version
        self.tipo = tipo

    def get_data_predict(self, model_periodo:int, periodo:int):
        """
        Obtiene los datos de predicción para un periodo dado.
        
        Args:
            model_periodo (int): Periodo del modelo.
            periodo (int): Periodo de evaluación.
            
        Returns:
            pd.DataFrame: DataFrame con los datos de predicción.
        """
        
        data_model_predict = pd.read_parquet(
            self.input_predict_datastore/'test'/f"data_predict_{self.model_version}_{model_periodo}_{self.tipo}_{periodo}.parquet")
        
        return data_model_predict

    def get_data_target(self, periodo:int):
        """
        Obtiene los datos objetivo para un periodo dado.
        
        Args:
            periodo (int): Periodo de evaluación.
            
        Returns:
            pd.DataFrame: DataFrame con los datos objetivo.
        """
        data_model_target = pd.read_parquet(
            self.input_target_datastore/'test'/self.target_version/f"data_target_{self.tipo}_{periodo}.parquet")
        return data_model_target
    
    def get_data_evaluation(self, model_periodo:int, periodo:int):
        """
        Obtiene los datos de evaluación para un periodo dado.
        
        Args:
            model_periodo (int): Periodo del modelo.
            periodo (int): Periodo de evaluación.
            
        Returns:
            pd.DataFrame: DataFrame con los datos de evaluación.
        """
        on_cols = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']
        
        data_model_predict = self.get_data_predict(model_periodo, periodo)
        data_model_target = self.get_data_target(periodo)

        data_model_evaluation = join_target(data_model_predict, data_model_target, on_cols)

        return data_model_evaluation
    
    def upload_data_evaluation(
        self, model_periodo:int, model_version:str, periodo:int, df_model_evaluation:pd.DataFrame, 
        mode:str):
        """
        Sube los datos de evaluación al datastore.
        
        Args:
            model_periodo (int): Periodo del modelo.
            model_version (str): Versión del modelo.
            periodo (int): Periodo de evaluación.
            df_model_evaluation (pd.DataFrame): DataFrame con los datos de evaluación.
            mode (str): Modo de evaluación.
        """

        path_model = (self.output_evaluation_datastore/'test')

        path_model.mkdir(parents=True, exist_ok=True)
        
        df_model_evaluation.to_parquet(
            path_model/f"data_evaluation_{model_version}_{model_periodo}_{self.tipo}_{periodo}.parquet"
        )

        assert mode in ['prod', 'dev']
        
        if mode == 'dev':
            df_model_evaluation.to_parquet(
                path_model/f"data_evaluation_dev_{model_periodo}_{self.tipo}_{periodo}.parquet"
            )
        else:
            df_model_evaluation.to_parquet(
                path_model/f"data_evaluation_prod_{model_periodo}_{self.tipo}_{periodo}.parquet"
            )
    
    def get_precision(self, model_periodo:int, periodo:int):
        """
        Calcula la precisión del modelo para un periodo dado.
        
        Args:
            model_periodo (int): Periodo del modelo.
            periodo (int): Periodo de evaluación.
            
        Returns:
            float: Precisión del modelo.
        """
        
        df_forecast = self.get_data_predict(model_periodo, periodo)
        df_target = self.get_data_target(periodo)
        
        on_cols = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'HORARIO_ACTUAL']
        
        metrics = calculate_metrics(df_forecast, df_target, on_cols)

        return metrics['precision']

    
    def upload_response(self, scores:list):
        """
        Sube la respuesta al datastore.
        
        Args:
            scores (list): Lista de precisiones.
        """
        precision = sum(scores)/len(scores)
        
        threshold = treshold_tipos[self.tipo]

        if precision>=threshold:
            print(f"Precision is {precision}, which is above the threshold of {threshold}")
            response = 'approved'
        else:
            print(f"Precision is {precision}, which is below the threshold of {threshold}")
            response = 'rejected'
        
        # Guardar en archivo para recuperar desde GitHub Actions
        metric_file = self.output_evaluation_datastore / f"response_{self.tipo}.txt"
        
        # Asegurar que el directorio existe
        self.output_evaluation_datastore.mkdir(parents=True, exist_ok=True)
        
        with open(metric_file, "w") as f:
            f.write(str(response))
            
        print(f"Metric saved to {metric_file}")


def main(args):
    """
    Función principal para evaluar el modelo de continuidad regular.
    
    Args:
        args: Argumentos de la línea de comandos.
    """
    input_predict_datastore = args.input_predict_datastore
    input_target_datastore = args.input_target_datastore
    output_evaluation_datastore = args.output_evaluation_datastore
    target_version = args.target_version
    model_periodo = args.model_periodo
    model_version = args.model_version
    n_eval_periodos = args.n_eval_periodos

    periodo = args.periodo
    mode = args.mode
    with_tipo = args.with_tipo

    tipo = 'inicial_regular'
    eval_tipo = eval(with_tipo)

    if not eval_tipo:
        input_predict_datastore = f"{input_predict_datastore}/{tipo}"
        input_target_datastore = f"{input_target_datastore}/{tipo}"
        output_evaluation_datastore = f"{output_evaluation_datastore}/{tipo}"
    
    train_inicial = TrainInicial(
        input_predict_datastore,
        input_target_datastore,
        output_evaluation_datastore,
        target_version,
        model_version,
        tipo)

    tipo = train_inicial.tipo

    if periodo == -1:
        assert n_eval_periodos >= 1
        periodo = model_periodo
        periodos = get_ahead_n_periodos(model_periodo, n_eval_periodos)
    else:
        periodos = [periodo]

    scores = []
    for periodo in periodos:
        map_tipos = get_mapping_tipos(periodo)
        if map_tipos[tipo]:
            df_model_evaluation = train_inicial.get_data_evaluation(model_periodo,periodo)
            train_inicial.upload_data_evaluation(model_periodo, model_version, periodo, df_model_evaluation, mode)
            precision = train_inicial.get_precision(model_periodo, periodo)
            scores.append(precision)
    
    train_inicial.upload_response(scores)


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