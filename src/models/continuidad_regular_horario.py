"""Script para entrenar el modelo de continuidad regular horario."""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from utils_model import parse_args, get_mapping_tipos


class TrainContinuidadToHorario:
    """
    Clase para entrenar el modelo de continuidad regular horario.
    """
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
        self.tipo = tipo
    
    def apply_filter(self, df_train:pd.DataFrame):
        """
        Aplica filtros al dataframe de entrenamiento.
        """
        return df_train
    
    def get_data_train(self, periodo:int):
        """
        Obtiene los datos de entrenamiento para el periodo dado.
        """
        data_model_train = pd.read_parquet(
            self.input_feats_datastore/"train"/self.feats_version/f"data_feats_{self.tipo}_{periodo}.parquet")
        return data_model_train  
    
    def train_model(self, periodo:int):
        """
        Entrena el modelo de continuidad regular horario.
        """
        data_model_train = self.get_data_train(periodo)
        data_model_train = self.apply_filter(data_model_train)
        meses = [1, 2, 3]
        # granular = ['PERIODO_TARGET', 'SEDE', 'CURSO_ACTUAL','HORARIO_ACTUAL',
        #             'IDX', 'PE', 'VAC_ACAD_ESTANDAR']
        
        cat_features = ['NIVEL', 'LEVEL', 'IDX_CURSO', 'SEDE']
        target = 'FAC_ALUMNOS'

        if periodo % 100 in meses:
            num_features = ['FAC_ALUMNOS_LAG_12', 'FAC_ALUMNOS_ANTERIOR']
            x = num_features + cat_features
            y = target
            # numerical_columns = {'FAC_CLASES': ['FAC_CLASES_LAG_12', 'FAC_CLASES_ANTERIOR'],
            #                      'FAC_ALUMNOS': ['FAC_ALUMNOS_LAG_12', 'FAC_ALUMNOS_ANTERIOR']}
            data_model_train = data_model_train[
                (data_model_train['PERIODO_TARGET'] % 100).isin([periodo % 100])
                # (self.df_model['PERIODO_TARGET'] < periodo)
                # & 
            ].copy()
            
            X_train = data_model_train[x].copy()
            y_train = data_model_train[y].copy()

        else:
            num_features = ['FAC_ALUMNOS_ANTERIOR']
            x = num_features + cat_features
            y = target
            
            data_model_train = data_model_train[
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
        
            # X_train, y_train = data_train_01[x].copy(), data_train[y].copy()
            
        model.fit(X_train, y_train, cat_features=cat_features)
        
        return model

    def save_model(self, model, periodo:int):
        """
        Guarda el modelo entrenado.
        """
        
         #path_model = self.output_model_datastore/'test'/self.model_version
         # 1. Definimos la ruta
        path_model = self.output_model_datastore / "model"
        
         # 2. AGREGAR ESTA LÍNEA: Crea la carpeta físicamente en el disco
        path_model.mkdir(parents=True, exist_ok=True)
         # 3. Guardamos el modelo
        model.save_model(path_model / f"{self.tipo}_{periodo}.cbm")
              
       
        print(f"Modelo guardado exitosamente en: {path_model / f'{self.tipo}_{periodo}.cbm'}") 
        #print(f"Guardando modelo en: {path_model/f'{self.tipo}_{periodo}.cbm'}")
        
        # path_model.mkdir(parents=True, exist_ok=True)        
        # model.save_model(path_model/f"{self.tipo}_{periodo}.cbm")
        
        # print(f"Guardando modelo en: {path_model/f'{self.tipo}_{periodo}.cbm'}")



def main(args):
    """
    Función principal para entrenar el modelo de continuidad regular horario.
    """
    input_feats_datastore = args.input_feats_datastore
    output_model_datastore = args.output_model_datastore
    feats_version = args.feats_version
    model_version = args.model_version
    mode = args.mode
    with_tipo = args.with_tipo
    model_periodo = args.model_periodo

    eval_tipo = eval(with_tipo)
    
    tipo = 'continuidad_regular_horario'
    
    if not eval_tipo:
        output_model_datastore = f"{output_model_datastore}/{tipo}"
        input_feats_datastore = f"{input_feats_datastore}/{tipo}"
    
    train_continuidad_horario = TrainContinuidadToHorario(
        input_feats_datastore,
        output_model_datastore,
        feats_version,
        model_version,
        tipo
        )  
    
    mapping_tipos = get_mapping_tipos(model_periodo)
    
    if mapping_tipos[train_continuidad_horario.tipo]:
        model = train_continuidad_horario.train_model(model_periodo)
        print("Training for:", model_periodo)
        train_continuidad_horario.save_model(model, model_periodo)
    

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