"""Machine-learning risk prediction helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "models" / "leave_risk_model.pkl"
FEATURE_NAMES = [
    "current_attendance",
    "present_classes",
    "absent_classes",
    "total_classes",
    "remaining_classes",
    "weekly_subject_classes",
    "minimum_attendance",
    "planned_leave_classes",
    "days_remaining",
    "absence_rate",
    "future_attendance_after_leave",
]


def _fallback_risk(features: dict[str, float]) -> str:
    projected = features["present_classes"] / max(
        1, features["total_classes"] + features["planned_leave_classes"]
    ) * 100
    margin = projected - features["minimum_attendance"]
    if margin < 0:
        return "HIGH"
    if margin < 5 or features["planned_leave_classes"] > features["remaining_classes"] * 0.5:
        return "MEDIUM"
    return "LOW"


def predict_risk(features: dict[str, float]) -> str:
    if not MODEL_PATH.exists():
        return _fallback_risk(features)
    try:
        model = joblib.load(MODEL_PATH)
        values = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)
        return str(model.predict(pd.DataFrame(values, columns=FEATURE_NAMES))[0])
    except (OSError, ValueError, KeyError):
        return _fallback_risk(features)


def predict_details(features: dict[str, float]) -> dict[str, Any]:
    """Return model output without presenting probabilities as certainty."""
    if not MODEL_PATH.exists():
        risk = _fallback_risk(features)
        return {"risk_level": risk, "probabilities": {risk: 1.0}}
    try:
        model = joblib.load(MODEL_PATH)
        values = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
        probabilities = model.predict_proba(values)[0]
        classes = [str(label) for label in model.classes_]
        return {
            "risk_level": str(model.predict(values)[0]),
            "probabilities": {label: round(float(probability), 6) for label, probability in zip(classes, probabilities)},
        }
    except (OSError, ValueError, KeyError):
        risk = _fallback_risk(features)
        return {"risk_level": risk, "probabilities": {risk: 1.0}}


def features_for_subject(
    current_attendance: float,
    present_classes: int,
    absent_classes: int,
    remaining_classes: int,
    weekly_subject_classes: int,
    minimum_attendance: float,
    planned_leave_classes: int,
    days_remaining: int,
    future_attendance_after_leave: float | None = None,
) -> dict[str, float]:
    total = present_classes + absent_classes
    future_total = total + planned_leave_classes
    projected = (present_classes / future_total * 100) if future_total else 0.0
    return {
        "current_attendance": current_attendance,
        "present_classes": present_classes,
        "absent_classes": absent_classes,
        "total_classes": total,
        "remaining_classes": remaining_classes,
        "weekly_subject_classes": weekly_subject_classes,
        "minimum_attendance": minimum_attendance,
        "planned_leave_classes": planned_leave_classes,
        "days_remaining": days_remaining,
        "absence_rate": absent_classes / total if total else 0.0,
        "future_attendance_after_leave": projected if future_attendance_after_leave is None else future_attendance_after_leave,
    }


def predict_subject_risks(
    config: dict[str, Any], subjects: dict[str, dict[str, Any]], planned_leave: dict[str, int] | None = None
) -> dict[str, dict[str, Any]]:
    planned_leave = planned_leave or {}
    days_remaining = len(config["remaining_working_days"])
    weekly = {
        subject: sum(day.count(subject) for day in config["timetable"].values())
        for subject in config["subjects"]
    }
    return {
        subject: predict_details(features_for_subject(
            float(values["attendance_percentage"]),
            int(values["present"]),
            int(values["absent"]),
            int(values["remaining_classes"]),
            weekly[subject],
            config["minimum_attendance"],
            planned_leave.get(subject, 0),
            days_remaining,
        ))
        for subject, values in subjects.items()
    }
