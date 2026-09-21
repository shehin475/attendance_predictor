"""Deterministic attendance and leave calculations."""
from __future__ import annotations

from datetime import date, timedelta
from math import floor
from typing import Any, Iterable

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def parse_date(value: str, field_name: str = "date") -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO date string (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid date in YYYY-MM-DD format") from exc


def date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    minimum = payload.get("minimum_attendance")
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not 0 <= minimum <= 100:
        raise ValueError("minimum_attendance must be between 0 and 100")

    calendar = payload.get("academic_calendar")
    if not isinstance(calendar, dict):
        raise ValueError("academic_calendar is required")
    start = parse_date(calendar.get("start_date"), "academic_calendar.start_date")
    end = parse_date(calendar.get("end_date"), "academic_calendar.end_date")
    if end < start:
        raise ValueError("academic_calendar.end_date cannot be before start_date")

    working_weekdays = payload.get("working_weekdays")
    if not isinstance(working_weekdays, list) or not working_weekdays:
        raise ValueError("working_weekdays must be a non-empty list")
    if any(day not in WEEKDAYS for day in working_weekdays) or len(set(working_weekdays)) != len(working_weekdays):
        raise ValueError("working_weekdays contains an invalid or duplicate weekday")

    def parse_holidays(field: str) -> set[date]:
        values = payload.get(field, [])
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list of dates")
        return {parse_date(value, field) for value in values}

    holidays = parse_holidays("holidays") | parse_holidays("special_holidays")

    timetable = payload.get("timetable")
    if not isinstance(timetable, dict):
        raise ValueError("timetable is required")
    subjects = payload.get("subjects")
    if not isinstance(subjects, dict) or not subjects:
        raise ValueError("subjects must be a non-empty object")
    for subject_name, attendance in subjects.items():
        if not isinstance(subject_name, str) or not subject_name.strip():
            raise ValueError("subject names cannot be empty")
        if not isinstance(attendance, dict):
            raise ValueError(f"attendance for {subject_name} must be an object")
        for key in ("present", "absent"):
            value = attendance.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{subject_name}.{key} must be a non-negative integer")
    for weekday, scheduled in timetable.items():
        if weekday not in WEEKDAYS:
            raise ValueError(f"invalid timetable weekday: {weekday}")
        if not isinstance(scheduled, list):
            raise ValueError(f"timetable.{weekday} must be a list")
        for subject in scheduled:
            if not isinstance(subject, str) or not subject.strip():
                raise ValueError("timetable subject names cannot be empty")
            if subject not in subjects:
                raise ValueError(f"timetable subject {subject} is not present in subjects")

    as_of = parse_date(payload.get("as_of_date", date.today().isoformat()), "as_of_date")
    return {
        **payload,
        "minimum_attendance": float(minimum),
        "start_date": start,
        "end_date": end,
        "holidays_set": holidays,
        "working_weekdays_set": set(working_weekdays),
        "as_of_date_value": as_of,
    }


def working_days(config: dict[str, Any], start: date | None = None, end: date | None = None) -> list[date]:
    first = start or config["start_date"]
    last = end or config["end_date"]
    if first > last:
        return []
    return [
        current
        for current in date_range(first, last)
        if current.strftime("%A") in config["working_weekdays_set"] and current not in config["holidays_set"]
    ]


def subject_attendance(present: int, absent: int) -> dict[str, float | int]:
    total = present + absent
    percentage = (present / total * 100) if total else 0.0
    return {
        "present": present,
        "absent": absent,
        "total_classes": total,
        "attendance_percentage": percentage,
    }


def maximum_safe_leave(present: int, absent: int, minimum_attendance: float) -> int:
    if minimum_attendance <= 0:
        return 10**9
    if present == 0:
        return 0
    required_ratio = minimum_attendance / 100
    allowance = present / required_ratio - present - absent
    return max(0, floor(allowance + 1e-10))


def future_class_counts(config: dict[str, Any], days: Iterable[date] | None = None) -> dict[str, int]:
    counts = {subject: 0 for subject in config["subjects"]}
    selected_days = days if days is not None else working_days(config, max(config["start_date"], config["as_of_date_value"]))
    for current in selected_days:
        for subject in config["timetable"].get(current.strftime("%A"), []):
            counts[subject] += 1
    return counts


def _round(value: float) -> float:
    return round(value, 2)


def attendance_after_leave(config: dict[str, Any], missed_classes: dict[str, int]) -> dict[str, dict[str, float | int]]:
    result = {}
    for subject, values in config["subjects"].items():
        present = values["present"]
        absent = values["absent"] + missed_classes.get(subject, 0)
        result[subject] = subject_attendance(present, absent)
    return result


def simulate_dates(config: dict[str, Any], leave_days: Iterable[date]) -> dict[str, Any]:
    missed = {subject: 0 for subject in config["subjects"]}
    affected = []
    for current in leave_days:
        scheduled = config["timetable"].get(current.strftime("%A"), [])
        affected.extend(scheduled)
        for subject in scheduled:
            missed[subject] += 1
    after = attendance_after_leave(config, missed)
    below = [
        subject
        for subject, values in after.items()
        if values["attendance_percentage"] + 1e-12 < config["minimum_attendance"]
    ]
    return {
        "safe": not below,
        "subjects_affected": sorted(set(affected)),
        "classes_missed": {subject: count for subject, count in missed.items() if count},
        "attendance_after_leave": {subject: {**values, "attendance_percentage": _round(values["attendance_percentage"])} for subject, values in after.items()},
        "subjects_below_minimum": below,
    }


def subject_results(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    remaining = future_class_counts(config)
    result = {}
    for subject, values in config["subjects"].items():
        attendance = subject_attendance(values["present"], values["absent"])
        safe_leave = maximum_safe_leave(values["present"], values["absent"], config["minimum_attendance"])
        result[subject] = {
            **attendance,
            "attendance_percentage": _round(attendance["attendance_percentage"]),
            "remaining_classes": remaining[subject],
            "maximum_safe_leave": min(safe_leave, remaining[subject]),
        }
    return result


def safe_leave_dates(config: dict[str, Any]) -> list[dict[str, Any]]:
    days = working_days(config, max(config["start_date"], config["as_of_date_value"]))
    results = []
    for current in days:
        simulation = simulate_dates(config, [current])
        results.append({
            "date": current.isoformat(),
            "subjects": simulation["subjects_affected"],
            "status": "SAFE" if simulation["safe"] else "NOT_RECOMMENDED",
        })
    return results


def risk_from_subjects(config: dict[str, Any], results: dict[str, dict[str, Any]]) -> str:
    if not results:
        return "LOW"
    if any(item["attendance_percentage"] < config["minimum_attendance"] for item in results.values()):
        return "HIGH"
    ratios = [item["maximum_safe_leave"] / max(1, item["remaining_classes"]) for item in results.values()]
    if min(ratios) < 0.2:
        return "HIGH"
    if min(ratios) < 0.5:
        return "MEDIUM"
    return "LOW"
