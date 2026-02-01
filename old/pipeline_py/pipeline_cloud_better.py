from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input, Output, command
from azure.ai.ml.constants import AssetTypes
import json
import os
import sys
import subprocess

# Add src to path just in case
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

def build_pipeline_better_func(periodos, model_periodo, script_types, compute, environment, versions, new_version):
    @dsl.pipeline(description="Pipeline Cloud Better (Replicating Local Better)")
    def pipeline_better_func(ult_periodo, n_periodos, all_periodos, 
                           platinum_version_pool,
                           exp_model_name, exp_predict_name):
        
        # ---------------------------- ETL (Dev/New Version Logic) -----------------------------------
        # In PS1: --platinum_version $new_version.platinum_version.dev
        platinum_v_dev = new_version.get('platinum_version', {}).get('dev', 'v1')

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
            display_name="ETL"
        )

        etl_step = etl_component(
            input_datastore=Input(type=AssetTypes.URI_FOLDER, path="azureml://datastores/rawdata/paths/rawdata"),
            platinum_version=platinum_v_dev,
            ult_periodo=ult_periodo,
            n_periodos=n_periodos,
            all_periodos=all_periodos
        )
        etl_step.compute = compute
        etl_step.name = "etl_step"
        etl_step.outputs.output_datastore = Output(
            type=AssetTypes.URI_FOLDER, 
            path="azureml://datastores/ml_data/paths/platinumdata",
            mode="rw_mount")
        
        # This output maps to platinumdata for subsequent steps
        platinum_data_output = etl_step.outputs.output_datastore

        # ---------------------------- Features & Models (Model Periodo - New Version) -------------------------
        mapping_tipos_model = get_mapping_tipos(model_periodo)
        
        # Store model steps to enforce dependency
        model_steps = {}

        for tipo in script_types:
            if mapping_tipos_model.get(tipo):
                # Dev Versions
                v_feats_dev = new_version.get('feats_version', {}).get(tipo, {}).get('dev', "v1")
                v_target_dev = new_version.get('target_version', {}).get(tipo, {}).get('dev', "v1")

                # --- Features Train (Dev) ---
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
                    platinum_version=platinum_v_dev,
                    feats_version=v_feats_dev,
                    target_version=v_target_dev,
                    ult_periodo=ult_periodo
                )
                feat_step.compute = compute
                feat_step.name = feat_step_name
                
                # Output paths to global store
                feat_step.outputs.output_feats_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/ml_data/paths/features")
                # Target not strictly needed to map if not used by others, but good practice if concurrent
                # feat_step.outputs.output_target_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/ml_data/paths/target")

                # --- Models Train (Dev) ---
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
                    feats_version=v_feats_dev
                )
                model_step.compute = compute
                model_step.name = model_step_name
                model_step.outputs.output_model_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/ml_data/paths/models")
                
                model_steps[tipo] = model_step


        # ---------------------------- Loop (Champion & Dev Paths) ----------------------------
        for periodo in periodos:
            mapping_tipos = get_mapping_tipos(periodo)
            
            for tipo in script_types:
                if mapping_tipos.get(tipo):
                    
                    # === CHAMPION PATH ===
                    v_feats_champ = versions.get('feats_version', {}).get(tipo, {}).get('champion', "v1")
                    v_target_champ = versions.get('target_version', {}).get(tipo, {}).get('champion', "v1")
                    platinum_v_champ = versions.get('platinum_version', {}).get('champion', "v1")

                    # 1. Features (Champ)
                    feat_champ_name = f"feat_champ_{tipo}_{periodo}"
                    feat_champ_step = feat_component(
                        input_datastore=platinum_data_output, # or generic input if champion uses different logic? assuming same ETL output
                        platinum_version=platinum_v_champ,
                        feats_version=v_feats_champ,
                        target_version=v_target_champ,
                        ult_periodo=ult_periodo
                    )
                    feat_champ_step.compute = compute
                    feat_champ_step.name = feat_champ_name
                    feat_champ_step.display_name = f"Feat Champ {tipo} {periodo}"
                    feat_champ_step.outputs.output_feats_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/ml_data/paths/features")

                    # 2. Predict (Champ)
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

                    pred_champ_name = f"pred_champ_{tipo}_{periodo}"
                    pred_champ_step = predict_component(
                        input_feats_datastore=feat_champ_step.outputs.output_feats_datastore,
                        input_target_datastore=feat_champ_step.outputs.output_target_datastore,
                        feats_version=v_feats_champ,
                        target_version=v_target_champ,
                        experiment_name=exp_predict_name,
                        dummy_input=None # Champion path uses registered model, no dependency on current training
                    )
                    pred_champ_step.compute = compute
                    pred_champ_step.name = pred_champ_name
                    pred_champ_step.display_name = f"Pred Champ {tipo} {periodo}"
                    pred_champ_step.outputs.output_predict_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/ml_data/paths/predict")

                    # 3. Eval (Champ)
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
                    
                    eval_champ_name = f"eval_champ_{tipo}_{periodo}"
                    eval_champ_step = eval_component(
                         input_predict_datastore=pred_champ_step.outputs.output_predict_datastore,
                         input_target_datastore=feat_champ_step.outputs.output_target_datastore,
                         target_version=v_target_champ
                    )
                    eval_champ_step.compute = compute
                    eval_champ_step.name = eval_champ_name
                    eval_champ_step.display_name = f"Eval Champ {tipo} {periodo}"
                    eval_champ_step.outputs.output_evaluation_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/ml_data/paths/evaluation")


                    # === DEV PATH === (New Version)
                    v_feats_dev = new_version.get('feats_version', {}).get(tipo, {}).get('dev', "v1")
                    v_target_dev = new_version.get('target_version', {}).get(tipo, {}).get('dev', "v1")
                    platinum_v_dev = new_version.get('platinum_version', {}).get('dev', "v1")

                    # 1. Features (Dev)
                    feat_dev_name = f"feat_dev_{tipo}_{periodo}"
                    feat_dev_step = feat_component(
                        input_datastore=platinum_data_output,
                        platinum_version=platinum_v_dev,
                        feats_version=v_feats_dev,
                        target_version=v_target_dev,
                        ult_periodo=ult_periodo
                    )
                    feat_dev_step.compute = compute
                    feat_dev_step.name = feat_dev_name
                    feat_dev_step.display_name = f"Feat Dev {tipo} {periodo}"
                    feat_dev_step.outputs.output_feats_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/ml_data/paths/features")

                    # 2. Predict (Dev) - MUST WAIT FOR MODEL TRAIN
                    # Prepare optional dependency input
                    dependency_input = None
                    if tipo in model_steps:
                         dependency_input = model_steps[tipo].outputs.output_model_datastore

                    pred_dev_name = f"pred_dev_{tipo}_{periodo}"
                    pred_dev_step = predict_component(
                        input_feats_datastore=feat_dev_step.outputs.output_feats_datastore,
                        input_target_datastore=feat_dev_step.outputs.output_target_datastore,
                        feats_version=v_feats_dev,
                        target_version=v_target_dev,
                        experiment_name=exp_predict_name,
                        dummy_input=dependency_input
                    )
                    pred_dev_step.compute = compute
                    pred_dev_step.name = pred_dev_name
                    pred_dev_step.display_name = f"Pred Dev {tipo} {periodo}"
                    pred_dev_step.outputs.output_predict_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/ml_data/paths/predict")

                    # 3. Eval (Dev)
                    eval_dev_name = f"eval_dev_{tipo}_{periodo}"
                    eval_dev_step = eval_component(
                         input_predict_datastore=pred_dev_step.outputs.output_predict_datastore,
                         input_target_datastore=feat_dev_step.outputs.output_target_datastore,
                         target_version=v_target_dev
                    )
                    eval_dev_step.compute = compute
                    eval_dev_step.name = eval_dev_name
                    eval_dev_step.display_name = f"Eval Dev {tipo} {periodo}"
                    eval_dev_step.outputs.output_evaluation_datastore = Output(type=AssetTypes.URI_FOLDER, path="azureml://datastores/ml_data/paths/evaluation")

        # ---------------------------- Compare & Update (Placeholder) ----------------------------
        # Logic to compare models and update champion
        # Note: Scripts src/model/{tipo}.py were requested but not found in src/models. 
        # Also placement after loop suggests comparing for last period?
        
        # for tipo in script_types:
        #     # Placeholder for Compare Step
        #     pass

    return pipeline_better_func

if __name__ == '__main__':
    # Configuration matches pipeline_local_better.ps1
    periodos = [202506, 202507, 202508, 202509, 202510, 202511]
    ult_periodo = max(periodos)
    model_periodo = min(periodos)
    n_periodos = "None"
    all_periodos = "True"
    
    script_types = ["continuidad_regular", "inicial_regular", "continuidad_estacional", "inicial_estacional"]
    
    environment = 'docker-image-plus-conda-example:2'
    compute = 'mldevci'
    
    # FETCH VERSIONS
    
    version_data = {}
    new_version_data = {}

    print("Fetching versions...")
    try:
        # Fetch current version
        res = subprocess.run(
            ["python", "./src/version/fetch_version.py", "--input_version_datastore", "./data/base_data/version"],
            capture_output=True, text=True, check=True
        )
        version_data = json.loads(res.stdout)
        
        # Fetch next version
        res_next = subprocess.run(
            ["python", "./src/version/fetch_next_version.py", "--input_version_datastore", "./data/base_data/version"],
            capture_output=True, text=True, check=True
        )
        new_version_data = json.loads(res_next.stdout)
        
        print("Versions fetched successfully.")
    except Exception as e:
        print(f"Warning: Could not fetch versions via subprocess ({e}). Using empty defaults.")
    
    # Initialize ML Client
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient.from_config(credential=credential)
    except Exception as e:
        print(f"Skipping MLClient init: {e}")
        ml_client = None

    pipeline_func = build_pipeline_better_func(
        periodos=periodos,
        model_periodo=model_periodo,
        script_types=script_types,
        compute=compute,
        environment=environment,
        versions=version_data,
        new_version=new_version_data
    )

    pipeline_job = pipeline_func(
        ult_periodo=ult_periodo,
        n_periodos=n_periodos,
        all_periodos=all_periodos,
        platinum_version_pool="v1", # Not directly used, versions passed in build
        exp_model_name="Train Models",
        exp_predict_name="Predict Models"
    )

    if ml_client:
        print("Submitting pipeline...")
        submitted_job = ml_client.jobs.create_or_update(
            pipeline_job, experiment_name="pipeline_better_cloud"
        )
        print(f"Pipeline submitted: {submitted_job.studio_url}")
