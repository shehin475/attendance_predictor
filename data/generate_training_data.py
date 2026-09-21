"""Generate reproducible, mathematically labelled synthetic training data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NUM_SAMPLES = 75_000
RANDOM_STATE = 42
OUTPUT_PATH = Path(__file__).resolve().parent / "training_data.csv"
FEATURE_NAMES = [
    "current_attendance", "present_classes", "absent_classes", "total_classes",
    "remaining_classes", "weekly_subject_classes", "minimum_attendance",
    "planned_leave_classes", "days_remaining", "absence_rate",
    "future_attendance_after_leave",
]
OUTPUT_COLUMNS = FEATURE_NAMES + ["leave_risk"]


def generate_data(num_samples: int = NUM_SAMPLES, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    stages = rng.choice([0, 1, 2, 3], size=num_samples, p=[0.22, 0.32, 0.30, 0.16])
    ranges = np.array([[10, 30], [30, 60], [60, 100], [100, 150]])
    total_classes = rng.integers(ranges[stages, 0], ranges[stages, 1] + 1)
    requirements = rng.choice([65, 70, 75, 80, 85, 90], size=num_samples, p=[0.08, 0.16, 0.36, 0.22, 0.13, 0.05])
    profile = rng.choice(["high", "borderline", "low", "very_low"], size=num_samples, p=[0.35, 0.30, 0.25, 0.10])
    target_attendance = np.empty(num_samples)
    target_attendance[profile == "high"] = rng.uniform(95, 100, (profile == "high").sum())
    target_attendance[profile == "borderline"] = requirements[profile == "borderline"] + rng.uniform(-3, 3, (profile == "borderline").sum())
    target_attendance[profile == "low"] = rng.uniform(50, 70, (profile == "low").sum())
    target_attendance[profile == "very_low"] = rng.uniform(20, 49.9, (profile == "very_low").sum())
    present_classes = np.floor(total_classes * np.clip(target_attendance / 100, 0, 1)).astype(int)
    absent_classes = total_classes - present_classes
    weekly = rng.integers(1, 7, num_samples)
    remaining = np.clip(((4 - stages) * 18 + rng.integers(-8, 15, num_samples)), 1, 70)
    planned_leave = np.minimum(rng.poisson(3, num_samples), rng.integers(0, 16, num_samples))
    high_leave = rng.random(num_samples) < 0.08
    planned_leave[high_leave] = rng.integers(10, 21, high_leave.sum())
    days_remaining = np.maximum(1, np.ceil(remaining / np.maximum(weekly / 5, 0.2)).astype(int) + rng.integers(-3, 4, num_samples))
    total_after_leave = present_classes + absent_classes + planned_leave
    future_attendance = np.divide(present_classes * 100, total_after_leave, out=np.zeros(num_samples, dtype=float), where=total_after_leave != 0)
    margin = future_attendance - requirements
    labels = np.select([margin < 0, margin < 5], ["HIGH", "MEDIUM"], default="LOW")
    return pd.DataFrame({
        "current_attendance": np.divide(present_classes * 100, total_classes, out=np.zeros(num_samples, dtype=float), where=total_classes != 0).round(4),
        "present_classes": present_classes,
        "absent_classes": absent_classes,
        "total_classes": total_classes,
        "remaining_classes": remaining,
        "weekly_subject_classes": weekly,
        "minimum_attendance": requirements,
        "planned_leave_classes": planned_leave,
        "days_remaining": days_remaining,
        "absence_rate": np.divide(absent_classes, total_classes, out=np.zeros(num_samples, dtype=float), where=total_classes != 0).round(6),
        "future_attendance_after_leave": future_attendance.round(4),
        "leave_risk": labels,
    }, columns=OUTPUT_COLUMNS)


def main() -> None:
    data = generate_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(data):,} synthetic samples at {OUTPUT_PATH}")
    print(data["leave_risk"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
