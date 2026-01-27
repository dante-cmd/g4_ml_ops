from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import argparse
# import mlflow
from mlflow.tracking import MlflowClient


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_feats_datastore", dest='input_feats_datastore',
                        type=str)
    parser.add_argument("--input_target_datastore", dest='input_target_datastore',
                        type=str)
    parser.add_argument("--output_predict_datastore", dest='output_predict_datastore',
                        type=str)
    parser.add_argument("--feats_version", dest='feats_version',
                        type=str)
    parser.add_argument("--target_version", dest='target_version',
                        type=str)
    parser.add_argument("--periodo", dest='periodo',
                        type=int)
    parser.add_argument("--experiment_name", dest='experiment_name',
                        type=str)
    parser.add_argument("--model_periodo", dest='model_periodo',
                        type=int)
    parser.add_argument("--dummy_input", dest='dummy_input',
                        type=str, required=False)
    
    # parse args
    args = parser.parse_args()

    # return args
    return args

# Optional: Set tracking URI if not using default
# mlflow.set_tracking_uri("http://your-mlflow-server:5000")



# model_name = "YourModelName"
# tag_key = "stage"
# tag_value = "Production"  # or any value like "staging", "latest", etc.

# Get all versions of the model

def get_dev_version(model_name) -> str:
    
    client = MlflowClient("http://127.0.0.1:5000")
    
    versions = client.get_registered_model(model_name)
    # versions = client.search_model_versions(f"name='{model_name}'")

    # # Sort by version number (descending) and pick the latest
    # latest_version = max(versions, key=lambda v: int(v.version))
    # latest_version.version

    return versions.aliases['dev']