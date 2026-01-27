
# Define parameters
$periodos = @(202506, 202507, 202508, 202509, 202510, 202511)
$ult_periodo = ($periodos | Measure-Object -Maximum).Maximum
$model_periodo=($periodos | Measure-Object -Minimum).Minimum
$n_periodos = "None"
$all_periodos = "True"
# ---------------------------
# Load versions from versions.json
$versionsJson = Get-Content -Path "./versions.json" -Raw | ConvertFrom-Json
$raw_version = $versionsJson.raw_version.champion
$platinum_version = $versionsJson.platinum_version.champion
# Note: feats_version, target_version, and model_version are now tipo-specific
# They will be retrieved per tipo from $versionsJson
# ---------------------------


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
    "--input_raw_datastore", "./data/base_data/rawdata/",
    "--output_platinum_datastore", "./data/ml_data/platinumdata/"
    "--platinum_version", "$platinum_version",
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
foreach ($tipo in $script_types) {
    $mapping_tipos = Get-MappingTipos -periodo $model_periodo
    if ($mapping_tipos[$tipo]) {
        # Get tipo-specific versions
        $feats_version = $versionsJson.feats_version.$tipo.champion
        $target_version = $versionsJson.target_version.$tipo.champion
        
        $feats_args = @(
            "--input_datastore", "./data/ml_data/platinumdata/",
            "--output_feats_datastore", "./data/ml_data/features/",
            "--output_target_datastore", "./data/ml_data/target/",
            "--platinum_version", "$platinum_version",
            "--feats_version", "$feats_version",
            "--target_version", "$target_version",
            "--periodo", "$model_periodo",
            "--ult_periodo", "$ult_periodo"
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
        $feats_version = $versionsJson.feats_version.$tipo.champion
        
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
            $feats_version = $versionsJson.feats_version.$tipo.champion
            $target_version = $versionsJson.target_version.$tipo.champion
            
            $feats_args = @(
                "--input_datastore", "./data/ml_data/platinumdata/$platinum_version/",
                "--output_feats_datastore", "./data/ml_data/features/",
                "--output_target_datastore", "./data/ml_data/target/",
                "--feats_version", "$feats_version",
                "--target_version", "$target_version",
                "--periodo", "$periodo",
                "--ult_periodo", "$ult_periodo"
            )
            
            $script = "./src/features/$tipo.py"
            Run-PythonScript -ScriptPath $script -ScriptArgs $feats_args
        }
    }

    # ----------------------------- Predict ----------------------------
    foreach ($tipo in $script_types) {
        if ($mapping_tipos[$tipo]) {
            # Get tipo-specific versions
            $feats_version = $versionsJson.feats_version.$tipo.champion
            $target_version = $versionsJson.target_version.$tipo.champion
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
}
