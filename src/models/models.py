from utils_model import parse_args, get_mapping_tipos
from loader import Loader
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
import argparse
from unidecode import unidecode

from inicial_regular import Inicial as InicialRegular
from continuidad_regular import Continuidad as ContinuidadRegular
from continuidad_regular import ContinuidadToHorario as ContinuidadToHorarioRegular
from inicial_estacional import Inicial as InicialEstacional
from continuidad_estacional import Continuidad as ContinuidadEstacional


def main(args):
    input_feats_datastore = args.input_feats_datastore
    experiment_name = args.experiment_name
    feats_version = args.feats_version
    model_periodo = args.model_periodo
    
    # listening to port 5000
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # 1. Initialize client
    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    
    mlflow.set_experiment(experiment_name)
    
    
    train_inicial = TrainInicial(
        input_feats_datastore, 
        feats_version,
        client)
    
    mapping_tipos = get_mapping_tipos(model_periodo)
    
    if mapping_tipos[train_inicial.tipo]:
        model = train_inicial.train_model(model_periodo)
        train_inicial.register_model(model, model_periodo)
    

if __name__ == "__main__":
    
    args = parse_args()
    main(args)
