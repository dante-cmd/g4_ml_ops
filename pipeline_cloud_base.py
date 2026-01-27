from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.ai.ml.constants import AssetTypes
import json
import os
import sys

# Add src to path to import version fetcher if needed, or just define helpers
sys.path.append("./src")

def get_mapping_tipos(periodo: int) -> dict:
    mod = periodo % 100
    tipos = {
        "inicial_estacional": False,
        "continuidad_estacional": False,
        "inicial_regular": True,
        "continuidad_regular": True
    }
    if mod == 1:
        tipos["inicial_estacional"] = True
    elif mod == 2:
        tipos["inicial_estacional"] = True
        tipos["continuidad_estacional"] = True
    return tipos

def build_pipeline_base_func(periodos, model_periodo, script_types, compute, environment, versions):
    @dsl.pipeline(description="Pipeline Cloud Base (Replicating PowerShell)")
    def pipeline_base_func(ult_periodo, n_periodos, all_periodos, 
                           platinum_version, feats_version_default, target_version_default,
                           exp_model_name, exp_predict_name):
        
        # ---------------------------- ETL -----------------------------------
        etl_component = command(
            code="./src/data",
            command="python etl.py \
                --input_datastore ${{inputs.input_datastore}} \
                --output_datastore ${{outputs.output_datastore}} \
                --platinum_version ${{inputs.platinum_version}} \
                --ult_periodo ${{inputs.ult_periodo}} \
                --n_periodos ${{inputs.n_periodos}} \
                --all_periodos ${{inputs.all_periodos}}",
            inputs={
                "input_datastore": Input(type=AssetTypes.URI_FOLDER),
                "platinum_version": Input(type="string"),
                "ult_periodo": Input(type="integer"),
                "n_periodos": Input(type="string"),
                "all_periodos": Input(type="string")
            },
            outputs={
                "output_datastore": Output(type=AssetTypes.URI_FOLDER)
            },
            environment=environment,
            display_name="etl_component"
        )

        etl_step = etl_component(
            input_datastore=Input(type=AssetTypes.URI_FOLDER, path="azureml://datastores/rawdata/paths/rawdata"),
            platinum_version=platinum_version,
            ult_periodo=ult_periodo,
            n_periodos=n_periodos,
            all_periodos=all_periodos
        )
        etl_step.compute = compute
        etl_step.name = "etl_step"
        etl_step.display_name = "ETL"

        platinum_data_output = etl_step.outputs.output_datastore

        # ---------------------------- Features & Models (Model Periodo) -------------------------
        # Logic: For model_periodo, run features then models
        
        mapping_tipos_model = get_mapping_tipos(model_periodo)
        
        # Store model steps to enforce dependency for predict steps later
        model_steps = {}

        for tipo in script_types:
            if mapping_tipos_model.get(tipo):
                # Versions for this tipo
                # In PS1: $version.feats_version.$tipo.champion
                # We use the passed versions dict or default
                v_feats = versions.get('feats_version', {}).get(tipo, {}).get('champion', "v1")
                v_target = versions.get('target_version', {}).get(tipo, {}).get('champion', "v1")

                # Features Step
                feat_step_name = f"feats_train_{tipo}"
                feat_component = command(
                    code="./src/features",
                    command=f"python {tipo}.py \
                        --input_datastore ${{{{inputs.input_datastore}}}} \
                        --output_feats_datastore ${{{{outputs.output_feats_datastore}}}} \
                        --output_target_datastore ${{{{outputs.output_target_datastore}}}} \
                        --platinum_version ${{{{inputs.platinum_version}}}} \
                        --feats_version ${{{{inputs.feats_version}}}} \
                        --target_version ${{{{inputs.target_version}}}} \
                        --periodo {model_periodo} \
                        --ult_periodo ${{{{inputs.ult_periodo}}}}",
                    inputs={
                        "input_datastore": Input(type=AssetTypes.URI_FOLDER),
                        "platinum_version": Input(type="string"),
                        "feats_version": Input(type="string"),
                        "target_version": Input(type="string"),
                        "ult_periodo": Input(type="integer")
                    },
                    outputs={
                        "output_feats_datastore": Output(type=AssetTypes.URI_FOLDER),
                        "output_target_datastore": Output(type=AssetTypes.URI_FOLDER)
                    },
                    environment=environment,
                    display_name=f"Feats Train {tipo}"
                )

                feat_step = feat_component(
                    input_datastore=platinum_data_output,
                    platinum_version=platinum_version,
                    feats_version=v_feats,
                    target_version=v_target,
                    ult_periodo=ult_periodo
                )
                feat_step.compute = compute
                feat_step.name = feat_step_name
                
                # Output paths (Global Stores)
                # Note: Azure ML will mount the OUTPUT of this step. 
                # Ideally we map this to the global datastore path so subsequent steps find it.
                # However, usually we pass the output object.
                feat_step.outputs.output_feats_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/platinumdata/paths/features")
                # feat_step.outputs.output_target_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/platinumdata/paths/target")

                # Models Step
                model_step_name = f"model_train_{tipo}"
                model_component = command(
                    code="./src/models",
                    command=f"python {tipo}.py \
                        --input_feats_datastore ${{{{inputs.input_feats_datastore}}}} \
                        --output_model_datastore ${{{{outputs.output_model_datastore}}}} \
                        --experiment_name ${{{{inputs.experiment_name}}}} \
                        --feats_version ${{{{inputs.feats_version}}}} \
                        --model_periodo {model_periodo}",
                    inputs={
                        "input_feats_datastore": Input(type=AssetTypes.URI_FOLDER),
                        "experiment_name": Input(type="string"),
                        "feats_version": Input(type="string")
                    },
                    outputs={
                        "output_model_datastore": Output(type=AssetTypes.URI_FOLDER)
                    },
                    environment=environment,
                    display_name=f"Model Train {tipo}"
                )

                model_step = model_component(
                    input_feats_datastore=feat_step.outputs.output_feats_datastore,
                    experiment_name=exp_model_name,
                    feats_version=v_feats
                )
                model_step.compute = compute
                model_step.name = model_step_name
                model_step.outputs.output_model_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/platinumdata/paths/models")
                
                model_steps[tipo] = model_step

        # ---------------------------- Loop (Features, Predict, Eval) ----------------------------
        for periodo in periodos:
            mapping_tipos = get_mapping_tipos(periodo)
            
            for tipo in script_types:
                if mapping_tipos.get(tipo):
                    v_feats = versions.get('feats_version', {}).get(tipo, {}).get('champion', "v1")
                    v_target = versions.get('target_version', {}).get(tipo, {}).get('champion', "v1")
                    
                    # Features Step (Inference)
                    feat_inf_name = f"feats_inf_{tipo}_{periodo}"
                    feat_inf_component = command(
                        code="./src/features",
                        command=f"python {tipo}.py \
                        --input_datastore ${{{{inputs.input_datastore}}}} \
                        --output_feats_datastore ${{{{outputs.output_feats_datastore}}}} \
                        --output_target_datastore ${{{{outputs.output_target_datastore}}}} \
                        --platinum_version ${{{{inputs.platinum_version}}}} \
                        --feats_version ${{{{inputs.feats_version}}}} \
                        --target_version ${{{{inputs.target_version}}}} \
                        --periodo {periodo} \
                        --ult_periodo ${{{{inputs.ult_periodo}}}}",
                        inputs={
                            "input_datastore": Input(type=AssetTypes.URI_FOLDER),
                            "platinum_version": Input(type="string"),
                            "feats_version": Input(type="string"),
                            "target_version": Input(type="string"),
                            "ult_periodo": Input(type="integer")
                        },
                        outputs={
                            "output_feats_datastore": Output(type=AssetTypes.URI_FOLDER),
                            "output_target_datastore": Output(type=AssetTypes.URI_FOLDER)
                        },
                        environment=environment,
                        display_name=f"Feats Inf {tipo} {periodo}"
                    )
                    
                    feat_inf_step = feat_inf_component(
                        input_datastore=platinum_data_output,
                        platinum_version=platinum_version,
                        feats_version=v_feats,
                        target_version=v_target,
                        ult_periodo=ult_periodo
                    )
                    feat_inf_step.compute = compute
                    feat_inf_step.name = feat_inf_name
                    feat_inf_step.outputs.output_feats_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/platinumdata/paths/features")
                    feat_inf_step.outputs.output_target_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/platinumdata/paths/target")

                    # Predict Step
                    predict_name = f"predict_{tipo}_{periodo}"
                    predict_component = command(
                        code="./src/predict",
                        command=f"python {tipo}.py \
                        --input_feats_datastore ${{{{inputs.input_feats_datastore}}}} \
                        --input_target_datastore ${{{{inputs.input_target_datastore}}}} \
                        --output_predict_datastore ${{{{outputs.output_predict_datastore}}}} \
                        --feats_version ${{{{inputs.feats_version}}}} \
                        --target_version ${{{{inputs.target_version}}}} \
                        --periodo {periodo} \
                        --model_periodo {model_periodo} \
                        --experiment_name ${{{{inputs.experiment_name}}}} \
                        --dummy_input ${{{{inputs.dummy_input}}}}",
                        inputs={
                            "input_feats_datastore": Input(type=AssetTypes.URI_FOLDER),
                            "input_target_datastore": Input(type=AssetTypes.URI_FOLDER),
                            "feats_version": Input(type="string"),
                            "target_version": Input(type="string"),
                            "experiment_name": Input(type="string"),
                            "dummy_input": Input(type=AssetTypes.URI_FOLDER, optional=True)
                        },
                        outputs={
                            "output_predict_datastore": Output(type=AssetTypes.URI_FOLDER)
                        },
                        environment=environment,
                        display_name=f"Predict {tipo} {periodo}"
                    )
                    
                    # Prepare optional dependency input
                    dependency_input = None
                    if tipo in model_steps:
                         dependency_input = model_steps[tipo].outputs.output_model_datastore

                    predict_step = predict_component(
                        input_feats_datastore=feat_inf_step.outputs.output_feats_datastore,
                        input_target_datastore=feat_inf_step.outputs.output_target_datastore,
                        feats_version=v_feats,
                        target_version=v_target,
                        experiment_name=exp_predict_name,
                        dummy_input=dependency_input
                    )
                    predict_step.compute = compute
                    predict_step.name = predict_name
                    predict_step.outputs.output_predict_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/platinumdata/paths/predict")

                    # Enforce dependency via data input instead of run_after
                    # if tipo in model_steps:
                    #     predict_step.run_after(model_steps[tipo])

                    # Evaluation Step
                    eval_name = f"eval_{tipo}_{periodo}"
                    eval_component = command(
                         code="./src/evaluation",
                         command=f"python {tipo}.py \
                        --input_predict_datastore ${{{{inputs.input_predict_datastore}}}} \
                        --input_target_datastore ${{{{inputs.input_target_datastore}}}} \
                        --output_evaluation_datastore ${{{{outputs.output_evaluation_datastore}}}} \
                        --target_version ${{{{inputs.target_version}}}} \
                        --periodo {periodo} \
                        --model_periodo {model_periodo}",
                        inputs={
                            "input_predict_datastore": Input(type=AssetTypes.URI_FOLDER),
                            "input_target_datastore": Input(type=AssetTypes.URI_FOLDER),
                            "target_version": Input(type="string")
                        },
                        outputs={
                            "output_evaluation_datastore": Output(type=AssetTypes.URI_FOLDER)
                        },
                        environment=environment,
                        display_name=f"Eval {tipo} {periodo}"
                    )
                    
                    eval_step = eval_component(
                        input_predict_datastore=predict_step.outputs.output_predict_datastore,
                        # Pass target from features step or directly?
                        # Using feat_inf_step outputs again
                        input_target_datastore=feat_inf_step.outputs.output_target_datastore,
                        target_version=v_target
                    )
                    eval_step.compute = compute
                    eval_step.name = eval_name
                    eval_step.outputs.output_evaluation_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/platinumdata/paths/evaluation")

    return pipeline_base_func

if __name__ == '__main__':
    # Configuration matches pipeline_local_base.ps1 parameters roughly
    periodos = [202506, 202507, 202508, 202509, 202510, 202511]
    ult_periodo = max(periodos)
    model_periodo = min(periodos)
    n_periodos = "None"
    all_periodos = "True"
    
    script_types = ["continuidad_regular", "inicial_regular", "continuidad_estacional", "inicial_estacional"]
    
    environment = 'docker-image-plus-conda-example:2'
    compute = 'mldevci'
    
    # Try to fetch version.json if exists or simulate
    # In a real scenario, we might want to run the python script here or load the file
    pass_versions = {}
    try:
        # Assuming we can run the fetch script or read the file if it exists locally
        # For this file generation, I will leave it as an empty dict or load if file exists
        # In cloud execution, the pipeline construction happens locally (runner)
        pass 
        # Example of loading: 
        # import subprocess
        # output = subprocess.check_output(["python", "src/version/fetch_version.py", "--input_version_datastore", "./data/base_data/version"])
        # pass_versions = json.loads(output)
    except:
        print("Could not fetch versions, using defaults")
    
    # Initialize ML Client
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient.from_config(credential=credential)
    except Exception as e:
        print(f"Skipping MLClient init (requires auth): {e}")
        ml_client = None

    pipeline_func = build_pipeline_base_func(
        periodos=periodos,
        model_periodo=model_periodo,
        script_types=script_types,
        compute=compute,
        environment=environment,
        versions=pass_versions
    )

    pipeline_job = pipeline_func(
        ult_periodo=ult_periodo,
        n_periodos=n_periodos,
        all_periodos=all_periodos,
        platinum_version="v1", # Default or from fetch
        feats_version_default="v1",
        target_version_default="v1",
        exp_model_name="Train Models",
        exp_predict_name="Predict Models"
    )

    if ml_client:
        print("Submitting pipeline...")
        submitted_job = ml_client.jobs.create_or_update(
            pipeline_job, experiment_name="pipeline_base_cloud"
        )
        print(f"Pipeline submitted: {submitted_job.studio_url}")
