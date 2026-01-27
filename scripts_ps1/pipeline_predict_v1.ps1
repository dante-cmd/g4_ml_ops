
# Define parameters
$periodos = @(202506, 202507, 202508, 202509, 202510, 202511)
$ult_periodo = ($periodos | Measure-Object -Maximum).Maximum
$model_periodo=($periodos | Measure-Object -Minimum).Minimum
$n_periodos = "None"
$all_periodos = "True"
$raw_version = "v1"
$platinum_version = "v1"
$features_version = "v1"
$target_version = "v1"
# $target_version = "v1"
$model_version = "v1"
$exp_model_name = "Train Models"
$exp_predict_name = "Predict Models"

# Helper function to run python scripts
function Run-PythonScript {
    param (
        [string]$ScriptPath,
        [string[]]$ScriptArgs
    )

    Write-Host "----------------------------------------------------------------"
    Write-Host "Executing: python $ScriptPath $ScriptArgs"
    Write-Host "----------------------------------------------------------------"
    
    & python $ScriptPath @ScriptArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Execution failed for $ScriptPath with exit code $LASTEXITCODE"
        # Uncomment the line below if you want the pipeline to stop on error
        # exit $LASTEXITCODE 
    }
}

# Function to determine which types of training/processing to run based on the period
function Get-MappingTipos {
    param (
        [int]$periodo
    )

    $tipos = @{
        "inicial_estacional"     = $false
        "continuidad_estacional" = $false
        "inicial_regular"        = $true
        "continuidad_regular"    = $true
    }

    $mod = $periodo % 100

    if ($mod -eq 1) {
        $tipos["inicial_estacional"] = $true
    }
    elseif ($mod -eq 2) {
        $tipos["inicial_estacional"] = $true
        $tipos["continuidad_estacional"] = $true
    }
    # else (default case) matches the default initialization above

    return $tipos
}

# --------------------------------------------------------------------
# ---------------------------- ETL -----------------------------------
# --------------------------------------------------------------------

$etl_script = "./src/data/etl.py"
$etl_args = @(
    "--input_datastore", "./data/base_data/rawdata/$raw_version/",
    "--output_datastore", "./data/ml_data/platinumdata/$platinum_version/",
    "--ult_periodo", "$ult_periodo",
    "--n_periodos", "$n_periodos",
    "--all_periodos", "$all_periodos"
)

Run-PythonScript -ScriptPath $etl_script -ScriptArgs $etl_args


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------

$script_types = @("continuidad_regular", "inicial_regular", "continuidad_estacional", "inicial_estacional")

# ---------------------------- Features for Model Periodo -------------------------
$feats_args = @(
        "--input_datastore", "./data/ml_data/platinumdata/$platinum_version/",
        "--output_feats_train_datastore", "./data/ml_data/features/$features_version/train/",
        "--output_feats_test_datastore", "./data/ml_data/features/$features_version/test/",
        "--output_target_test_datastore", "./data/ml_data/target/$target_version/test/",
        "--periodo", "$model_periodo",
        "--ult_periodo", "$ult_periodo"
    )

foreach ($tipo in $script_types) {
    $mapping_tipos = Get-MappingTipos -periodo $model_periodo
    if ($mapping_tipos[$tipo]) {
        $script = "./src/features/$tipo.py"
        Run-PythonScript -ScriptPath $script -ScriptArgs $feats_args
    }
}

# ----------------------------- Models for Model Periodo -----------------------------
$models_args = @(
        "--input_feats_train_datastore", "./data/ml_data/features/$features_version/train/",
        "--experiment_name", "$exp_model_name",
        "--model_periodo", "$model_periodo"
    )

foreach ($tipo in $script_types) {
    $mapping_tipos = Get-MappingTipos -periodo $model_periodo
    if ($mapping_tipos[$tipo]) {
        $script = "./src/models/$tipo.py"
        Run-PythonScript -ScriptPath $script -ScriptArgs $models_args
    }
}

# --------------------------------------------------------------------
# ----------------------- Features & Models --------------------------
# --------------------------------------------------------------------

foreach ($periodo in $periodos) {
    Write-Host "`nProcessing Period: $periodo"
    
    $mapping_tipos = Get-MappingTipos -periodo $periodo

    # ---------------------------- Features ----------------------------
    $feats_args = @(
        "--input_datastore", "./data/ml_data/platinumdata/$platinum_version/",
        "--output_feats_train_datastore", "./data/ml_data/features/$features_version/train/",
        "--output_feats_test_datastore", "./data/ml_data/features/$features_version/test/",
        "--output_target_test_datastore", "./data/ml_data/target/$target_version/test/",
        "--periodo", "$periodo",
        "--ult_periodo", "$ult_periodo"
    )

    foreach ($tipo in $script_types) {
        if ($mapping_tipos[$tipo]) {
            $script = "./src/features/$tipo.py"
            Run-PythonScript -ScriptPath $script -ScriptArgs $feats_args
        }
    }

    # ----------------------------- Predict ----------------------------
    $predict_args = @(
        "--input_feats_test_datastore", "./data/ml_data/features/$features_version/test/",
        "--input_target_test_datastore", "./data/ml_data/target/$target_version/test/",
        "--output_predict_test_datastore", "./data/ml_data/predict/$model_version/test/",
        "--experiment_name", "$exp_predict_name",
        "--periodo", "$periodo",
        "--model_periodo", "$model_periodo"
    )

    foreach ($tipo in $script_types) {
        if ($mapping_tipos[$tipo]) {
            $script = "./src/predict/$tipo.py"
            Run-PythonScript -ScriptPath $script -ScriptArgs $predict_args
        }
    }
}
