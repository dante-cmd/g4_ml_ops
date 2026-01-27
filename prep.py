# src/components/prep/prep.py
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()

    # Lee el CSV de entrada
    df = pd.read_csv(args.input_data)

    # Nombre de la columna objetivo según tu CSV
    target_col = "Diabetic"

    # Split estratificado
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[target_col],
    )

    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path / "train.csv", index=False)
    test_df.to_csv(test_path / "test.csv", index=False)


if __name__ == "__main__":
    main()