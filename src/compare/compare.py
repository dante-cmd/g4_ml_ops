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