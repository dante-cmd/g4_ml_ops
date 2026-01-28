
# ----------------------------- Define parameters -----------------------------
# Periodos, ult_periodo, model_periodo, n_periodos, all_periodos
$periodos = @(202506, 202507, 202508, 202509, 202510, 202511)
$ult_periodo = ($periodos | Measure-Object -Maximum).Maximum
$model_periodo=($periodos | Measure-Object -Minimum).Minimum
$n_periodos = "None"
$all_periodos = "True"

# Experiment names
$exp_model_name = "Train Models"
$exp_predict_name = "Predict Models"

$script_types = @("continuidad_regular", "inicial_regular", "continuidad_estacional", "inicial_estacional")

# Fetch version
$output = python .\src\version\fetch_version.py --input_version_datastore "./data/base_data/version"
$version = $output | ConvertFrom-Json

# Fetch next version
# $output = python .\src\version\fetch_next_version.py --input_version_datastore "./data/base_data/version"
# $new_version = $output | ConvertFrom-Json

# Write-Host $version.model_version.continuidad_regular.champion       #"Version: $($version.platinum_version.champion)"
# Write-Host $new_version.model_version.continuidad_regular.champion   #"Version: $($version.platinum_version.champion)"


# before to merge to main

# ----------------------------- Helper functions -----------------------------

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

# ---------------------------- ETL -----------------------------------

$etl_script = "./src/data/etl.py"
$etl_args = @(
    "--input_datastore", "./data/base_data/rawdata/",
    "--output_datastore", "./data/ml_data/platinumdata/"
    "--platinum_version", $version.platinum_version.champion,
    "--ult_periodo", "$ult_periodo",
    "--n_periodos", "$n_periodos",
    "--all_periodos", "$all_periodos"
)

Run-PythonScript -ScriptPath $etl_script -ScriptArgs $etl_args


# ---------------------------- Features for Model Periodo -------------------------
foreach ($tipo in $script_types) {
    $mapping_tipos = Get-MappingTipos -periodo $model_periodo
    if ($mapping_tipos[$tipo]) {
        # Get tipo-specific versions
        $feats_version = $version.feats_version.$tipo.champion
        $target_version = $version.target_version.$tipo.champion
        
        $feats_args = @(
            "--input_datastore", "./data/ml_data/platinumdata/",
            "--output_feats_datastore", "./data/ml_data/features/",
            "--output_target_datastore", "./data/ml_data/target/",
            "--platinum_version", $version.platinum_version.champion,
            "--feats_version", $feats_version,
            "--target_version", $target_version,
            "--periodo", $model_periodo,
            "--ult_periodo", $ult_periodo
        )
        
        $script = "./src/features/$tipo.py"
        Run-PythonScript -ScriptPath $script -ScriptArgs $feats_args
    }
}


# ----------------------------- Models for Model Periodo -----------------------------
foreach ($tipo in $script_types) {
    $mapping_tipos = Get-MappingTipos -periodo $model_periodo
    if ($mapping_tipos[$tipo]) {
        # Get tipo-specific version
        $feats_version = $version.feats_version.$tipo.champion
        
        $models_args = @(
            "--input_feats_datastore", "./data/ml_data/features/",
            "--feats_version", "$feats_version",
            "--experiment_name", "$exp_model_name",
            "--model_periodo", "$model_periodo"
        )
        
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
    foreach ($tipo in $script_types) {
        if ($mapping_tipos[$tipo]) {
            # Get tipo-specific versions
            $feats_version = $version.feats_version.$tipo.champion
            $target_version = $version.target_version.$tipo.champion
            
            $feats_args = @(
            "--input_datastore", "./data/ml_data/platinumdata/",
            "--output_feats_datastore", "./data/ml_data/features/",
            "--output_target_datastore", "./data/ml_data/target/",
            "--platinum_version", $version.platinum_version.champion,
            "--feats_version", $feats_version,
            "--target_version", $target_version,
            "--periodo", $periodo,
            "--ult_periodo", $ult_periodo
        )
            
            $script = "./src/features/$tipo.py"
            Run-PythonScript -ScriptPath $script -ScriptArgs $feats_args
        }
    }

    # ----------------------------- Predict ----------------------------
    foreach ($tipo in $script_types) {
        if ($mapping_tipos[$tipo]) {
            # Get tipo-specific versions
            $feats_version = $version.feats_version.$tipo.champion
            $target_version = $version.target_version.$tipo.champion
            # $model_version = $versionsJson.model_version.$tipo.champion
            
            $predict_args = @(
                "--input_feats_datastore", "./data/ml_data/features/",
                "--input_target_datastore", "./data/ml_data/target/",
                "--output_predict_datastore", "./data/ml_data/predict/",
                "--feats_version", "$feats_version",
                "--target_version", "$target_version",
                # "--model_version", "$model_version",
                "--periodo", "$periodo",
                "--model_periodo", "$model_periodo",
                "--experiment_name", "$exp_predict_name"
            )
            
            $script = "./src/predict/$tipo.py"
            Run-PythonScript -ScriptPath $script -ScriptArgs $predict_args
        }
    }

    # ----------------------------- Evaluation ----------------------------
    foreach ($tipo in $script_types) {
        if ($mapping_tipos[$tipo]) {
            $target_version = $version.target_version.$tipo.champion
            $eval_args = @(
                "--input_predict_datastore", "./data/ml_data/predict/",
                "--input_target_datastore", "./data/ml_data/target/",
                "--output_evaluation_datastore", "./data/ml_data/evaluation/",
                "--target_version", "$target_version",
                "--periodo", "$periodo",
                "--model_periodo", "$model_periodo"
                )
            $script = "./src/evaluation/$tipo.py"
            Run-PythonScript -ScriptPath $script -ScriptArgs $eval_args
        }
    }
}
