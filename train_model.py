"""Train and persist the leave-risk classifier."""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from model import FEATURE_NAMES, MODEL_PATH

DATA_PATH = Path(__file__).resolve().parent / "data" / "training_data.csv"
REAL_DATA_PATH = Path(__file__).resolve().parent / "data" / "real_training_data.csv"
METRICS_PATH = Path(__file__).resolve().parent / "models" / "model_metrics.json"


def load_training_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    if REAL_DATA_PATH.exists() and REAL_DATA_PATH.stat().st_size > 1:
        real = pd.read_csv(REAL_DATA_PATH)
        if not real.empty:
            if "future_attendance_after_leave" not in real:
                real["future_attendance_after_leave"] = real["attendance_after_leave"]
            data = pd.concat([data, real], ignore_index=True, sort=False)
    missing = set(FEATURE_NAMES + ["leave_risk"]) - set(data.columns)
    if missing:
        raise ValueError(f"training dataset is missing columns: {sorted(missing)}")
    return data[FEATURE_NAMES + ["leave_risk"]].dropna()


def train() -> None:
    data = load_training_data()
    features = data[FEATURE_NAMES]
    labels = data["leave_risk"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    classifier = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1,
        min_samples_leaf=2,
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    metrics = {
        "model": "RandomForestClassifier",
        "version": "1.0",
        "training_samples": int(len(data)),
        "test_samples": int(len(x_test)),
        "features": FEATURE_NAMES,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=classifier.classes_).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "feature_importance": {name: float(value) for name, value in zip(FEATURE_NAMES, classifier.feature_importances_)},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
