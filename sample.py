import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--model_periodo", type=int, required=True)
    parser.add_argument("--ult_periodo_data", type=int, required=True)
    parser.add_argument("--n_eval_periodos", type=int, required=True)
    parser.add_argument("--periodo", type=int, required=True)
    parser.add_argument("--with_tipo", type=str, required=True)
    parser.add_argument("--platinum_version", type=str, required=True)
    parser.add_argument("--feats_version", type=str, required=True)
    parser.add_argument("--target_version", type=str, required=True)
    parser.add_argument("--model_version", type=str, required=True)
    args = parser.parse_args()
    print(args)
