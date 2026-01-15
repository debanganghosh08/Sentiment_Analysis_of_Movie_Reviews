# src/evaluate.py

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate(model, X, y, name="Model"):
    preds = model.predict(X)

    print(f"\n{name} Results")
    print("-" * 40)
    print("Accuracy:", accuracy_score(y, preds))
    print("\nClassification Report:")
    print(classification_report(y, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, preds))
