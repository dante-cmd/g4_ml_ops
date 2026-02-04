from utils_compare import parse_args, get_mapping_tipos, get_ahead_n_periodos
import pandas as pd
import numpy as np
from pathlib import Path


class Compare:
    def __init__(self, 
        input_evaluation_inicial_datastore:str, 
        input_evaluation_continuidad_horario_datastore:str, 
        output_compare_datastore:str, 
        n_eval_periodos:int, 
        model_periodo:int, 
        model_current_version:str, 
        model_version:str, 
        periodo:int
        ):
        
        self.input_evaluation_inicial_datastore = Path(input_evaluation_inicial_datastore)
        self.input_evaluation_continuidad_horario_datastore = Path(input_evaluation_continuidad_horario_datastore)
        self.output_compare_datastore = Path(output_compare_datastore)
        
        self.n_eval_periodos = n_eval_periodos
        self.model_periodo = model_periodo
        self.model_current_version = model_current_version
        self.model_version = model_version
        self.periodo = periodo
    
    def get_score_evaluation(self, path_file:Path, periodo:int, model_version:str, tipo:str):
        
        data_model_predict = pd.read_parquet(
            path_file # self.input_evaluation_inicial_datastore
            /'test'
            /f"data_evaluation_{model_version}_{self.model_periodo}_{tipo}_{periodo}.parquet")
        
        print(f"Data loaded from {path_file /'test'/f'data_evaluation_{model_version}_{self.model_periodo}_{tipo}_{periodo}.parquet'}")
        filtro =  ((data_model_predict['CANT_CLASES_PREDICT'] == 0) & 
                    (data_model_predict['CANT_CLASES'] == 0))

        data_model_predict_01 = data_model_predict[~filtro].copy()

        equal_clases = data_model_predict_01['CANT_CLASES_PREDICT'] == data_model_predict_01['CANT_CLASES']
        score = float(np.mean(np.where(equal_clases, 1, 0)))
        
        return score

    def get_average_score(self, periodos:list[int], tipo:str, model_version:str, path_file:Path):
        scores = []
        for periodo in periodos:
            score = self.get_score_evaluation(path_file, periodo, model_version, tipo)
            scores.append(score)
        
        return sum(scores)/len(scores)

    def upload_response(self):

        if self.periodo == -1:
            assert self.n_eval_periodos >= 1
            # periodo = self.model_periodo
            periodos = get_ahead_n_periodos(self.model_periodo, self.n_eval_periodos)
        else:
            periodos = [self.periodo]

        tipos = {'inicial_regular':self.input_evaluation_inicial_datastore,
                  'continuidad_regular_horario':self.input_evaluation_continuidad_horario_datastore}
    
        evaluation_scores = []

        for tipo in tipos:
            current_score = self.get_average_score(periodos, tipo, self.model_current_version, tipos[tipo])
            
            print("model_current_version, model_version", self.model_current_version, self.model_version)
            print(f"Current score for {tipo}: {current_score}")
            
            new_score = self.get_average_score(periodos, tipo, self.model_version, tipos[tipo])
            print(f"New score for {tipo}: {new_score}")
            if new_score > current_score:
                evaluation_scores.append(True)
            else:
                evaluation_scores.append(False)
        
        if all(evaluation_scores):
            print("New model is better than current model")
            response = 'approved'
        else:
            print("New model is not better than current model")
            response = 'rejected'
        
        # Guardar en archivo para recuperar desde GitHub Actions
        metric_file = self.output_compare_datastore / f"response.txt"
        
        # Asegurar que el directorio existe
        self.output_compare_datastore.mkdir(parents=True, exist_ok=True)
        
        with open(metric_file, "w") as f:
            f.write(str(response))
            
        print(f"Metric saved to {metric_file}")

def main(args):
    input_evaluation_inicial_datastore = args.input_evaluation_inicial_datastore
    input_evaluation_continuidad_horario_datastore = args.input_evaluation_continuidad_horario_datastore
    output_compare_datastore = args.output_compare_datastore
    n_eval_periodos = args.n_eval_periodos
    model_periodo = args.model_periodo
    model_current_version = args.model_current_version
    model_version = args.model_version
    periodo = args.periodo
    # mode = args.mode
    with_tipo = args.with_tipo

    eval_tipo = eval(with_tipo)

    assert eval_tipo
    
    compare = Compare(
        input_evaluation_inicial_datastore=input_evaluation_inicial_datastore,
        input_evaluation_continuidad_horario_datastore=input_evaluation_continuidad_horario_datastore,
        output_compare_datastore=output_compare_datastore,
        n_eval_periodos=n_eval_periodos,
        model_periodo=model_periodo,
        model_current_version=model_current_version,
        model_version=model_version,
        periodo=periodo
    )

    compare.upload_response()


if __name__ == "__main__":
    args = parse_args()
    main(args)

# from mlflow.client import MlflowClient
# from pathlib import Path
# from utils_compare import get_mapping_tipos



# tipos = ['inicial_estacional', 'continuidad_estacional',
#          'inicial_regular', 'continuidad_regular']

# class Compare:
#     def __init__(self, 
#         input_evaluation_datastore:str, 
#         model_periodo:str):

#         self.input_evaluation_datastore  = Path(input_evaluation_datastore)
#         self.model_periodo = model_periodo
        
#         pass

# periodos = []





#     client = MlflowClient()

#     name_model = f"{tipo}_{model_periodo}"

#     version = client.get_registered_model(name_model)

#     model_dev_version = "v" + version['dev']
#     model_champion_version = "v" + version['champion']