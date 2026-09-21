from __future__ import annotations

from datetime import date

import pytest

from app import app, prepare_config
from leave_calculator import (
    future_class_counts,
    maximum_safe_leave,
    safe_leave_dates,
    simulate_dates,
    subject_attendance,
    validate_request,
    working_days,
)


BASE = {
    "minimum_attendance": 75,
    "academic_calendar": {"start_date": "2026-09-21", "end_date": "2026-09-30"},
    "holidays": ["2026-09-23"],
    "special_holidays": ["2026-09-24", "2026-09-23"],
    "working_weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    "timetable": {
        "Monday": ["Maths", "Maths", "DBMS"],
        "Tuesday": ["AI", "Maths"],
        "Wednesday": ["OS"],
        "Thursday": ["DBMS", "AI"],
        "Friday": ["OS", "DBMS"],
        "Saturday": ["Maths"],
    },
    "subjects": {
        "Maths": {"present": 35, "absent": 3},
        "DBMS": {"present": 31, "absent": 5},
        "AI": {"present": 28, "absent": 4},
        "OS": {"present": 34, "absent": 2},
    },
    "as_of_date": "2026-09-21",
}


def test_attendance_and_exact_leave_calculation():
    assert subject_attendance(35, 3)["attendance_percentage"] == pytest.approx(92.105263)
    assert maximum_safe_leave(35, 3, 75) == 8


def test_working_days_remove_weekends_and_both_holiday_types():
    config = validate_request(BASE)
    assert [day.isoformat() for day in working_days(config)] == ["2026-09-21", "2026-09-22", "2026-09-25", "2026-09-26", "2026-09-28", "2026-09-29", "2026-09-30"]


def test_repeated_timetable_entries_count_as_separate_classes():
    config = prepare_config(BASE)
    assert future_class_counts(config)["Maths"] == 7


def test_safe_dates_and_consecutive_leave():
    config = prepare_config(BASE)
    dates = safe_leave_dates(config)
    assert dates[0]["date"] == "2026-09-21"
    simulation = simulate_dates(config, working_days(config, date(2026, 9, 21), date(2026, 9, 22)))
    assert simulation["safe"] is True
    assert simulation["classes_missed"]["Maths"] == 3


def test_invalid_input():
    invalid = {**BASE, "minimum_attendance": 101}
    with pytest.raises(ValueError, match="minimum_attendance"):
        validate_request(invalid)


def test_flask_endpoints():
    client = app.test_client()
    assert client.get("/health").get_json() == {"status": "running"}
    response = client.post("/predict-leave", json=BASE)
    assert response.status_code == 200
    assert response.get_json()["status"] == "success"
    response = client.post("/calculate-attendance", json={"minimum_attendance": 75, "subjects": BASE["subjects"]})
    assert response.status_code == 200
    assert response.get_json()["subjects"]["Maths"]["maximum_safe_leave"] == 8
    response = client.post("/safe-leave-dates", json=BASE)
    assert response.status_code == 200
    assert response.get_json()["status"] == "success"
    response = client.post("/simulate-leave", json={**BASE, "leave_start": "2026-09-21", "leave_end": "2026-09-22"})
    assert response.status_code == 200
    assert response.get_json()["status"] == "success"
    response = client.post("/predict-leave", json=invalid_payload())
    assert response.status_code == 400


def test_dynamic_requirements_change_safe_leave_and_response_metadata():
    client = app.test_client()
    results = {}
    for requirement in (75, 80, 85, 90):
        response = client.post("/predict-leave", json={**BASE, "minimum_attendance": requirement})
        assert response.status_code == 200
        payload = response.get_json()
        results[requirement] = payload["subjects"]["Maths"]["maximum_safe_leave"]
        assert payload["minimum_attendance"] == requirement
        assert "probabilities" in payload["subjects"]["Maths"]
        assert payload["subjects"]["Maths"]["attendance_margin"] == pytest.approx(92.11 - requirement, abs=0.01)
    assert results[75] > results[80] > results[85] >= results[90]


def test_model_info_and_frontend_are_available():
    client = app.test_client()
    info = client.get("/model-info")
    assert info.status_code == 200
    model = info.get_json()
    assert model["training_samples"] >= 50_000
    assert model["model"] == "RandomForestClassifier"
    assert set(model["feature_importance"]) >= {"minimum_attendance", "future_attendance_after_leave"}
    assert client.get("/").status_code == 200


def invalid_payload():
    return {**BASE, "academic_calendar": {"start_date": "2026-10-01", "end_date": "2026-09-01"}}
