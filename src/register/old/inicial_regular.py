import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_inicial_path", type=str, help="Ruta donde está el modelo entrenado")
    parser.add_argument("--model_periodo", type=int, help="Periodo del modelo")
    parser.add_argument("--output_model_version", type=str, help="Ruta de salida para la versión del modelo")
    args = parser.parse_args()
    return args


def register_model_v1(workspace, model_path, model_name):
    """Register model using azureml v1 SDK"""
    from azureml.core import Model as V1Model
    
    # Register the model
    registered_model = V1Model.register(
        workspace=workspace,
        model_path=model_path,
        model_name=model_name,
        description=f"Modelo {model_name} registrado automáticamente"
    )
    
    print(f"Modelo registrado versión: {registered_model.version}")
    return str(registered_model.version)


def main(args):
    input_model_inicial_path = args.input_model_inicial_path
    model_periodo = args.model_periodo
    output_model_version = args.output_model_version
    
    tipo = 'inicial_regular'
    model_name = tipo + '_' + str(model_periodo)

    print(f"Registrando modelo desde: {input_model_inicial_path}")
    print(f"Registrando modelo con nombre: {model_name}")

    # Get workspace from Run context
    try:
        from azureml.core import Run
        run = Run.get_context()
        
        # Check if we're running in Azure ML (not offline run)
        if hasattr(run, 'experiment'):
            print("Running in Azure ML. Using workspace from Run context...")
            workspace = run.experiment.workspace
            
            # Copy model files to local directory
            local_model_dir = "model_temp"
            if os.path.exists(local_model_dir):
                shutil.rmtree(local_model_dir)
            
            print(f"Copying model from {input_model_inicial_path} to {local_model_dir}...")
            shutil.copytree(input_model_inicial_path, local_model_dir)
            
            # Register using v1 SDK from local path
            version = register_model_v1(workspace, local_model_dir, model_name)
            
            # Guardar la versión en el output path
            if output_model_version:
                with open(output_model_version, "w") as f:
                    f.write(version)
                print(f"Versión {version} guardada en {output_model_version}")
        else:
            raise Exception("Not running in Azure ML environment")
            
    except Exception as e:
        print(f"Error registering model: {e}")
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args)