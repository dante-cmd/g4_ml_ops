"""
Feature Engineering - Features
"""

from utils_feats import get_n_lags, get_training_periodos, filter_by_hora_atencion, parse_args, get_mapping_tipos
from parser_regular import Utils
from loader import Loader
from inicial_regular import Inicial as InicialRegular
from continuidad_regular import Continuidad as ContinuidadRegular
from continuidad_regular import ContinuidadToHorario as ContinuidadToHorarioRegular
from inicial_estacional import Inicial as InicialEstacional
from continuidad_estacional import Continuidad as ContinuidadEstacional


def main(args):
    # args = parse_args()
    input_datastore=args.input_datastore
    ult_periodo=args.ult_periodo
    
    # for Inicial
    output_feats_datastore = args.output_feats_datastore
    output_target_datastore = args.output_target_datastore
    platinum_version = args.platinum_version
    feats_version=args.feats_version
    target_version = args.target_version
    periodo=args.periodo

    mapping_tipos = get_mapping_tipos(periodo)
    
    loader = Loader(
        input_datastore, 
        platinum_version,
        ult_periodo)
    
    tablas = loader.fetch_all()
    
    # =================================================================
    # ======================== Inicial Regular ========================
    # =================================================================
    
    inicial_regular = InicialRegular(tablas)
    tipo_inicial_regular = inicial_regular.tipo
    
    if mapping_tipos[tipo_inicial_regular]:
        # Create directories if they don't exist
        feats_train_path = Path(output_feats_datastore) / tipo_inicial_regular / "train" / feats_version
        feats_test_path = Path(output_feats_datastore) / tipo_inicial_regular / "test" / feats_version
        target_test_path = Path(output_target_datastore) / tipo_inicial_regular / "test" / target_version
        
        feats_train_path.mkdir(parents=True, exist_ok=True)
        feats_test_path.mkdir(parents=True, exist_ok=True)
        target_test_path.mkdir(parents=True, exist_ok=True)

        inicial_regular.load_features(periodo, feats_version, output_feats_datastore)
        
        inicial_regular.load_target(periodo, target_version, output_target_datastore)
    else:
        print(f"No se generaron features para el periodo {periodo} y tipo {tipo}")

    # =================================================================
    # ======================== Continuidad Regular ====================
    # =================================================================
    
    continuidad_regular = ContinuidadRegular(tablas)
    tipo_continuidad_regular = continuidad_regular.tipo
    
    if mapping_tipos[tipo_continuidad_regular]:
        # Create directories if they don't exist
        feats_train_path = Path(output_feats_datastore) / tipo_continuidad_regular / "train" / feats_version
        feats_test_path = Path(output_feats_datastore) / tipo_continuidad_regular / "test" / feats_version
        target_test_path = Path(output_target_datastore) / tipo_continuidad_regular / "test" / target_version
        
        feats_train_path.mkdir(parents=True, exist_ok=True)
        feats_test_path.mkdir(parents=True, exist_ok=True)
        target_test_path.mkdir(parents=True, exist_ok=True)

        continuidad_regular.load_features(periodo, feats_version, output_feats_datastore)
        
        continuidad_regular.load_target(periodo, target_version, output_target_datastore)
    else:
        print(f"No se generaron features para el periodo {periodo} y tipo {tipo}")

    # =================================================================
    # ======================== Continuidad Regular Horario ============
    # =================================================================
    
    continuidad_to_horario_regular = ContinuidadToHorarioRegular(tablas)
    tipo_continuidad_to_horario_regular = continuidad_to_horario_regular.tipo
    
    if mapping_tipos[tipo_continuidad_to_horario_regular]:
        # Create directories if they don't exist
        feats_train_path = Path(output_feats_datastore) / tipo_continuidad_to_horario_regular / "train" / feats_version
        feats_test_path = Path(output_feats_datastore) / tipo_continuidad_to_horario_regular / "test" / feats_version
        target_test_path = Path(output_target_datastore) / tipo_continuidad_to_horario_regular / "test" / target_version
        
        feats_train_path.mkdir(parents=True, exist_ok=True)
        feats_test_path.mkdir(parents=True, exist_ok=True)
        target_test_path.mkdir(parents=True, exist_ok=True)

        continuidad_to_horario_regular.load_features(periodo, feats_version, output_feats_datastore)
        
        continuidad_to_horario_regular.load_target(periodo, target_version, output_target_datastore)
    else:
        print(f"No se generaron features para el periodo {periodo} y tipo {tipo}")


    # =================================================================
    # ======================== Inicial Estacional =====================
    # =================================================================
    
    inicial_estacional = InicialEstacional(tablas)
    tipo_inicial_estacional = inicial_estacional.tipo
    
    if mapping_tipos[tipo_inicial_estacional]:
        # Create directories if they don't exist
        feats_train_path = Path(output_feats_datastore) / tipo_inicial_estacional / "train" / feats_version
        feats_test_path = Path(output_feats_datastore) / tipo_inicial_estacional / "test" / feats_version
        target_test_path = Path(output_target_datastore) / tipo_inicial_estacional / "test" / target_version
        
        feats_train_path.mkdir(parents=True, exist_ok=True)
        feats_test_path.mkdir(parents=True, exist_ok=True)
        target_test_path.mkdir(parents=True, exist_ok=True)

        inicial_estacional.load_features(periodo, feats_version, output_feats_datastore)
        
        inicial_estacional.load_target(periodo, target_version, output_target_datastore)
    else:
        print(f"No se generaron features para el periodo {periodo} y tipo {tipo}")

    # =================================================================
    # ======================== Continuidad Estacional =================
    # =================================================================
    
    continuidad_estacional = ContinuidadEstacional(tablas)
    tipo_continuidad_estacional = continuidad_estacional.tipo
    
    if mapping_tipos[tipo_continuidad_estacional]:
        # Create directories if they don't exist
        feats_train_path = Path(output_feats_datastore) / tipo_continuidad_estacional / "train" / feats_version
        feats_test_path = Path(output_feats_datastore) / tipo_continuidad_estacional / "test" / feats_version
        target_test_path = Path(output_target_datastore) / tipo_continuidad_estacional / "test" / target_version
        
        feats_train_path.mkdir(parents=True, exist_ok=True)
        feats_test_path.mkdir(parents=True, exist_ok=True)
        target_test_path.mkdir(parents=True, exist_ok=True)

        continuidad_estacional.load_features(periodo, feats_version, output_feats_datastore)
        
        continuidad_estacional.load_target(periodo, target_version, output_target_datastore)
    else:
        print(f"No se generaron features para el periodo {periodo} y tipo {tipo}")


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