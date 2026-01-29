import argparse
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from utils_register import get_ml_client


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input_path", type=str, help="Ruta donde está el modelo entrenado")
    parser.add_argument("--model_periodo", type=int, help="Periodo del modelo")
    parser.add_argument("--output_model_version", type=str, help="Ruta de salida para la versión del modelo")
    args = parser.parse_args()
    return args


class Continuidad:
    def __init__(self, model_input_path, model_periodo:int, ml_client):
        self.model_input_path = model_input_path
        self.tipo = 'continuidad_regular'
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
            path=self.model_input_path,
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
        return registered_model.version


def main(args):
    
    model_input_path = args.model_input_path
    model_periodo = args.model_periodo
    output_model_version = args.output_model_version

    print(f"Registrando modelo desde: {model_input_path}")
    print(f"Registrando modelo con nombre: {model_periodo}")

    ml_client = get_ml_client()

    continuidad = Continuidad(model_input_path, model_periodo, ml_client)
    version = continuidad.register()

    # Guardar la versión en el output path
    if output_model_version:
        # Assuming output_model_version is a file path (uri_file)
        with open(output_model_version, "w") as f:
            f.write(version)
        print(f"Versión {version} guardada en {output_model_version}")


if __name__ == "__main__":
    args = parse_args()
    main(args)