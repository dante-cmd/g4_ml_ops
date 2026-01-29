import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from utils_register import parse_args, get_ml_client


class Inicial:
    def __init__(self, input_model_inicial_path, model_periodo:int, ml_client):
        self.input_model_inicial_path = input_model_inicial_path
        self.tipo = 'inicial_regular'
        self.model_periodo = model_periodo
        self.ml_client = ml_client

    def register(self):

        # 2. Lógica: ¿Existe ya el modelo?
        # Listamos los modelos que tengan ese nombre
        model_list = self.ml_client.models.list(name=self.tipo + '_' + str(self.model_periodo))
    
        # Verificamos si el iterador tiene al menos un elemento
        try:
            # Si esto funciona, es que ya existe al menos una versión
            _ = next(model_list)
            current_stage = "dev"
            print(f"El modelo '{self.tipo + '_' + str(self.model_periodo)}' YA EXISTE. Asignando etiqueta: {current_stage}")
        except StopIteration:
            # Si falla el next(), es que la lista estaba vacía
            current_stage = "champion"
            print(f"El modelo '{self.tipo + '_' + str(self.model_periodo)}' ES NUEVO. Asignando etiqueta: {current_stage}")

        # 3. Registrar con la etiqueta calculada
        model = Model(
            path=self.input_model_inicial_path,
            name=self.tipo + '_' + str(self.model_periodo),
            description=f"Modelo registrado automáticamente como {current_stage}",
            type=AssetTypes.MLFLOW_MODEL,
            tags={
                "stage": current_stage,
                "framework": "sklearn"
            }
        )

        # Registrar en Azure ML
        registered_model = self.ml_client.models.create_or_update(model)
        print(f"Modelo registrado versión: {registered_model.version}")


def main(args):
    
    input_model_inicial_path = args.input_model_inicial_path
    model_periodo = args.model_periodo

    print(f"Registrando modelo desde: {input_model_inicial_path}")
    print(f"Registrando modelo con nombre: {model_periodo}")

    ml_client = get_ml_client()

    inicial = Inicial(input_model_inicial_path, model_periodo, ml_client)
    inicial.register()

if __name__ == "__main__":
    args = parse_args()
    main(args)