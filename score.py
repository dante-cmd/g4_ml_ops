import os
import logging
import json
from catboost import CatBoostRegressor

def init():
    """Se ejecuta una vez cuando se levanta el contenedor"""
    global model
    # AZUREML_MODEL_DIR es la ruta automática donde Azure pone el modelo
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    
    # Buscamos cualquier archivo .cbm en la carpeta
    for file in os.listdir(model_path):
        if file.endswith(".cbm"):
            model = CatBoostRegressor()
            model.load_model(os.path.join(model_path, file))
            logging.info("Modelo CatBoost cargado exitosamente")
            break

def run(raw_data):
    """Se ejecuta en cada petición de predicción"""
    try:
        data = json.loads(raw_data)["data"]
        preds = model.predict(data)
        return {"prediction": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}