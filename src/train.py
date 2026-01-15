# src/train.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from joblib import dump

def train_logistic_regression(X_train, y_train, save_path):
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    dump(model, save_path)
    return model


def train_svm(X_train, y_train, save_path):
    model = LinearSVC()
    model.fit(X_train, y_train)
    dump(model, save_path)
    return model
