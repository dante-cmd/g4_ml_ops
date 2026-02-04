import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from utils_model import parse_args, get_mapping_tipos


class TrainContinuidad:
    def __init__(self,
                 input_feats_datastore:str,
                 output_model_datastore:str,
                 feats_version:str,
                 model_version:str,
                 tipo:str
                 ):
        
        self.input_feats_datastore = Path(input_feats_datastore)
        self.output_model_datastore = Path(output_model_datastore)
        self.feats_version = feats_version
        self.model_version = model_version
        self.tipo  = tipo

    def apply_filter(self, df_train:pd.DataFrame):
        return df_train
    
    def get_data_train(self, periodo:int):
        data_model_train = pd.read_parquet(
            self.input_feats_datastore/"train"/self.feats_version/f"data_feats_{self.tipo}_{periodo}.parquet")
        return data_model_train  
    
    def train_model(self, periodo:int):

        data_model_train = self.get_data_train(periodo)
        data_model_train = self.apply_filter(data_model_train)
        
        meses = [1, 2, 3]
        # granular = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL', 'IDX', 'PE', 'VAC_ACAD_ESTANDAR']
        
        cat_features = ['LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']

        if periodo % 100 in meses:
            # cat_features = ['LEVEL', 'IDX_CURSO', 'SEDE', 'FRECUENCIA', 'DURACION']
            # target = "PCT_ALUMNOS"
            # x = num_features + cat_features
            # y = target
            if periodo%100 in [1, 3]:
                num_features = [
                    'CANT_ALUMNOS_LAG_12', 
                    'CANT_ALUMNOS_ANTERIOR',
                    # 'DIFF_ALUMNOS_SPLY', 
                    'PCT_ALUMNOS_SPLY']
                target = "PCT_ALUMNOS"
                x = num_features + cat_features
                y = target
            else:
                num_features = [
                    # 'CANT_ALUMNOS_LAG_12', 
                    'CANT_ALUMNOS_ANTERIOR', 
                    # 'DIFF_ALUMNOS_SPLY', 
                    # 'PCT_ALUMNOS_SPLY'
                    ]
                target = "CANT_ALUMNOS"
                x = num_features + cat_features
                # x = cat_features
                y = target
            
            data_model_train = data_model_train[
                # (X_train['PERIODO_TARGET'] < periodo)
                (data_model_train['PERIODO_TARGET'] % 100).isin([periodo % 100])
            ].copy()
            
            X_train = data_model_train[x].copy()
            y_train = data_model_train[y].copy()

        else:
            num_features = ['CANT_ALUMNOS_ANTERIOR']
            target = "CANT_ALUMNOS"
            # x = num_features + cat_features
            x = cat_features
            y = target
            print("x", x)
            print("y", y)
            data_model_train = data_model_train[
                # (X_train['PERIODO_TARGET'] < periodo)
                ~(data_model_train['PERIODO_TARGET'] % 100).isin(meses)
            ].copy()
            X_train = data_model_train[x].copy()
            y_train = data_model_train[y].copy()
        
        # with mlflow.start_run():
        model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=False, # Set to True to see training progress,
        min_data_in_leaf=5,
        )

        model.fit(X_train, y_train, cat_features=cat_features)
        return model

    def save_model(self, model, periodo:int):
        
        path_model = self.output_model_datastore/'test'/self.model_version
        
        path_model.mkdir(parents=True, exist_ok=True)        
        model.save_model(path_model/f"{self.tipo}_{periodo}.cbm")
        
        print(f"Guardando modelo en: {path_model/f'{self.tipo}_{periodo}.cbm'}")


def main(args):
    input_feats_datastore = args.input_feats_datastore
    output_model_datastore = args.output_model_datastore
    feats_version = args.feats_version
    model_version = args.model_version
    mode = args.mode
    with_tipo = args.with_tipo
    model_periodo = args.model_periodo
    
    tipo = 'continuidad_regular'
    
    eval_tipo = eval(with_tipo)
        
    if not eval_tipo:
        output_model_datastore = f"{output_model_datastore}/{tipo}"
        input_feats_datastore = f"{input_feats_datastore}/{tipo}"
    
    train_continuidad = TrainContinuidad(
        input_feats_datastore,
        output_model_datastore,
        feats_version,
        model_version,
        tipo
        )
    
    mapping_tipos = get_mapping_tipos(model_periodo)
    
    if mapping_tipos[train_continuidad.tipo]:
        model = train_continuidad.train_model(model_periodo)
        print("Training for:", model_periodo)
        train_continuidad.save_model(model, model_periodo)


if __name__ == '__main__':
        # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")