# src/data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw_data(path):
    """
    Loads raw IMDb dataset.
    """
    df = pd.read_csv(path)
    return df


def encode_labels(df, label_col="sentiment"):
    """
    Encode sentiment labels:
    positive -> 1
    negative -> 0
    """
    df[label_col] = df[label_col].map({"positive": 1, "negative": 0})
    return df


def split_data(df, text_col="review", label_col="sentiment", seed=42):
    """
    Splits data into:
    - 80% train
    - 10% validation
    - 10% test
    """
    X = df[text_col]
    y = df[label_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir):
    """
    Saves splits as CSV files.
    """
    train_df = pd.DataFrame({"review": X_train, "sentiment": y_train})
    val_df = pd.DataFrame({"review": X_val, "sentiment": y_val})
    test_df = pd.DataFrame({"review": X_test, "sentiment": y_test})

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
