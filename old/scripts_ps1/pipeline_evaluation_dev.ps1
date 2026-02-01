
# Define parameters
$periodos = @(202506, 202507, 202508, 202509, 202510, 202511)
$versions = @("v1", "v2")
# $raw_version = "v1"
# $platinum_version = "v1"
# $features_version = "v1"
$target_version = "v1"
# $model_version = "v1"



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
# --------------------------------------------------------------------
# --------------------------------------------------------------------

$script_types = @("continuidad_regular", "inicial_regular", "continuidad_estacional", "inicial_estacional")


# --------------------------------------------------------------------
# ----------------------- Features & Models --------------------------
# --------------------------------------------------------------------

foreach ($periodo in $periodos) {
    Write-Host "`nProcessing Period: $periodo"
    
    $mapping_tipos = Get-MappingTipos -periodo $periodo

    # ---------------------------- Features ----------------------------
    foreach ($version in $versions) {

        $feats_args = @(
            "--input_predict_test_datastore", "./data/ml_data/predict/$version/test/",
            "--input_target_test_datastore", "./data/ml_data/target/$target_version/test/",
            "--output_evaluation_test_datastore", "./data/ml_data/evaluation/$version/test/",
            "--periodo", "$periodo"
        )

        foreach ($tipo in $script_types) {
            if ($mapping_tipos[$tipo]) {
                $script = "./src/evaluation/$tipo.py"
                Run-PythonScript -ScriptPath $script -ScriptArgs $feats_args
            }
        }
    }
}
