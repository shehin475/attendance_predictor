"""Flask API for student attendance and leave prediction."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any, Callable

from flask import Flask, jsonify, request
from flask_cors import CORS

from leave_calculator import (
    attendance_after_leave,
    future_class_counts,
    maximum_safe_leave,
    parse_date,
    risk_from_subjects,
    safe_leave_dates,
    simulate_dates,
    subject_attendance,
    subject_results,
    validate_request,
    working_days,
)
from model import MODEL_PATH, predict_subject_risks

METRICS_PATH = Path(__file__).resolve().parent / "models" / "model_metrics.json"
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)


def prepare_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = validate_request(payload)
    all_working = working_days(config)
    config["all_working_days"] = all_working
    config["remaining_working_days"] = working_days(
        config, max(config["start_date"], config["as_of_date_value"]), config["end_date"]
    )
    return config


def success_response(config: dict[str, Any]) -> dict[str, Any]:
    subjects = subject_results(config)
    risks = predict_subject_risks(config, subjects)
    for subject in subjects:
        current = subjects[subject]["attendance_percentage"]
        projected = current
        subjects[subject].update({
            "minimum_attendance": config["minimum_attendance"],
            "future_attendance_after_leave": projected,
            "attendance_margin": round(projected - config["minimum_attendance"], 2),
            "safe_leave": subjects[subject]["maximum_safe_leave"] > 0,
            "risk_level": risks[subject]["risk_level"],
            "prediction": risks[subject]["risk_level"],
            "probabilities": risks[subject]["probabilities"],
            "probability": risks[subject]["probabilities"].get(risks[subject]["risk_level"], 0.0),
        })
    return {
        "status": "success",
        "summary": {
            "total_calendar_days": (config["end_date"] - config["start_date"]).days + 1,
            "total_working_days": len(config["all_working_days"]),
            "remaining_working_days": len(config["remaining_working_days"]),
        },
        "subjects": subjects,
        "minimum_attendance": config["minimum_attendance"],
        "safe_leave_dates": safe_leave_dates(config),
        "warnings": [],
        "overall_risk": risk_from_subjects(config, subjects),
    }


def error_response(message: str, status_code: int = 400):
    return jsonify({"status": "error", "message": message}), status_code


def with_validation(handler: Callable[[dict[str, Any]], Any]):
    try:
        return handler(prepare_config(request.get_json(silent=True) or {}))
    except (TypeError, ValueError, KeyError) as exc:
        return error_response(str(exc))


@app.get("/health")
def health():
    return jsonify({"status": "running"})


@app.get("/")
def frontend():
    return app.send_static_file("index.html")


@app.get("/model-info")
def model_info():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return error_response("trained model metadata is unavailable; run the training commands first", 503)
    try:
        return jsonify({"status": "success", **json.loads(METRICS_PATH.read_text(encoding="utf-8"))})
    except (OSError, json.JSONDecodeError) as exc:
        return error_response(f"could not read model metadata: {exc}", 503)


@app.post("/predict-leave")
def predict_leave():
    return with_validation(lambda config: jsonify(success_response(config)))


@app.post("/calculate-attendance")
def calculate_attendance():
    payload = request.get_json(silent=True) or {}
    try:
        minimum = payload.get("minimum_attendance")
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not 0 <= minimum <= 100:
            raise ValueError("minimum_attendance must be between 0 and 100")
        subjects = payload.get("subjects")
        if not isinstance(subjects, dict) or not subjects:
            raise ValueError("subjects must be a non-empty object")
        result = {}
        for subject, values in subjects.items():
            if not isinstance(subject, str) or not subject.strip() or not isinstance(values, dict):
                raise ValueError("subjects must contain named attendance objects")
            present, absent = values.get("present"), values.get("absent")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (present, absent)):
                raise ValueError(f"{subject}.present and {subject}.absent must be non-negative integers")
            attendance = subject_attendance(present, absent)
            result[subject] = {
                **attendance,
                "attendance_percentage": round(attendance["attendance_percentage"], 2),
                "attendance_margin": round(attendance["attendance_percentage"] - float(minimum), 2),
                "maximum_safe_leave": maximum_safe_leave(present, absent, float(minimum)),
            }
        return jsonify({"status": "success", "minimum_attendance": minimum, "subjects": result})
    except (TypeError, ValueError, KeyError) as exc:
        return error_response(str(exc))


def parse_leave_range(config: dict[str, Any]) -> list[date]:
    leave_start = parse_date(config.get("leave_start"), "leave_start")
    leave_end = parse_date(config.get("leave_end"), "leave_end")
    if leave_end < leave_start:
        raise ValueError("leave_end cannot be before leave_start")
    if leave_start < config["start_date"] or leave_end > config["end_date"]:
        raise ValueError("leave period must be inside the academic calendar")
    return working_days(config, leave_start, leave_end)


@app.post("/simulate-leave")
def simulate_leave():
    def handle(config):
        simulation = simulate_dates(config, parse_leave_range(config))
        simulation["status"] = "success"
        return jsonify(simulation)

    return with_validation(handle)


@app.post("/safe-leave-dates")
def safe_dates():
    return with_validation(lambda config: jsonify({"status": "success", "safe_leave_dates": safe_leave_dates(config)}))


@app.errorhandler(404)
def not_found(_error):
    return error_response("endpoint not found", 404)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)