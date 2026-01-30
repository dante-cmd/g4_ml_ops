import argparse
import os
import shutil
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from utils_register import get_ml_client

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_inicial_path", type=str, help="Ruta donde está el modelo entrenado")
    parser.add_argument("--model_periodo", type=int, help="Periodo del modelo")
    parser.add_argument("--output_model_version", type=str, help="Ruta de salida para la versión del modelo")
    args = parser.parse_args()
    return args

def register_model_v2(ml_client, model_path, model_name, model_periodo):
    """Register model using azure-ai-ml v2 SDK"""
    
    # Check existing versions (for logging purposes)
    try:
        # Note: listing models in v2 returns an iterator
        existing_models = list(ml_client.models.list(name=model_name))
        if existing_models:
            # v2 models have 'version' as a string, usually integers '1', '2' etc.
            # We filter for integer-like versions to find max
            versions = [int(m.version) for m in existing_models if m.version.isdigit()]
            if versions:
                latest_version = max(versions)
                print(f"📊 Found {len(existing_models)} existing version(s) of '{model_name}'")
                print(f"   Latest version: {latest_version}")
            else:
                print(f"📊 Found existing versions but non-integer: {[m.version for m in existing_models]}")
        else:
            print(f"📊 No existing versions found for '{model_name}'")
    except Exception as e:
        print(f"⚠️  Could not check existing versions: {e}")

    # Prepare model entity
    # If version is not specified, Azure ML automatically increments it.
    model = Model(
        path=model_path,
        name=model_name,
        description=f"Modelo {model_name} registrado automáticamente con SDK v2",
        type=AssetTypes.CUSTOM_MODEL,
        tags={
            'modelo_periodo': str(model_periodo),
            'sdk': 'v2',
            'tipo': 'inicial_regular'
        }
    )
    
    print(f"Registrando modelo '{model_name}' con SDK v2...")
    registered_model = ml_client.models.create_or_update(model)
    
    print(f"✅ Modelo registrado: '{registered_model.name}' - Versión: {registered_model.version}")
    return str(registered_model.version)

def main(args):
    input_model_inicial_path = args.input_model_inicial_path
    model_periodo = args.model_periodo
    output_model_version = args.output_model_version
    
    tipo = 'inicial_regular'
    model_name = f"{tipo}_{model_periodo}"

    print(f"Registrando modelo desde: {input_model_inicial_path}")
    print(f"Registrando modelo con nombre: {model_name}")

    # Initialize MLClient (uses utils_register helper to use Run context)
    ml_client = get_ml_client()
            
    # Copy model files to local directory to avoid permission issues with mounted paths
    local_model_dir = "model_temp"
    if os.path.exists(local_model_dir):
        shutil.rmtree(local_model_dir)
    
    print(f"Copying model from {input_model_inicial_path} to {local_model_dir}...")
    shutil.copytree(input_model_inicial_path, local_model_dir)
    
    # Register using v2 SDK
    version = register_model_v2(ml_client, local_model_dir, model_name, model_periodo)
    
    # Guardar la versión en el output path
    # Ensure usage of output_model_version from args
    if output_model_version:
        with open(output_model_version, "w") as f:
            f.write(version)
        print(f"Versión {version} guardada en {output_model_version}")

if __name__ == "__main__":
    args = parse_args()
    main(args)