# Student Leave ML API

A Flask backend that calculates subject-wise attendance and determines how much leave a student can take without falling below a configurable minimum attendance percentage. Mathematical eligibility is deterministic; a RandomForest model is used only to classify leave risk.

## Features

- Academic calendar and inclusive calendar-day counts
- Configurable working weekdays, including Saturday classes
- Official and special holiday exclusion with duplicate removal
- Timetables with multiple classes of the same subject per day
- Exact subject attendance and maximum safe leave calculations
- Future subject class counts and safe leave date simulation
- Consecutive leave simulation
- Reproducible 75,000-row realistic synthetic dataset with mathematical labels
- Optional combination with `data/real_training_data.csv`
- RandomForest LOW/MEDIUM/HIGH risk classification with probabilities
- 80/20 stratified evaluation, confusion matrix, classification report, and feature importance
- CORS-enabled JSON API with validation
- Browser testing dashboard served at `/`

## Architecture

`leave_calculator.py` owns date, attendance, validation, and simulation logic. `model.py` owns feature construction and model inference. `app.py` adapts validated JSON requests to those services. `data/generate_training_data.py` creates reproducible synthetic data. `train_model.py` combines synthetic data with optional real historical rows, trains the persisted model, and writes `models/model_metrics.json`. `frontend/` contains the browser testing UI.

## Installation

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Train and run

```bash
python data/generate_training_data.py
python train_model.py
python app.py
```

The API runs at `http://localhost:5000`. If the model has not been trained, the API uses a deterministic fallback risk classifier so attendance endpoints remain available; training is recommended for production.

## Endpoints

- `GET /health`
- `POST /predict-leave`: full calculation, future classes, safe dates, authoritative eligibility, risk, and probabilities
- `POST /calculate-attendance`: `{ "minimum_attendance": 75, "subjects": { ... } }`
- `POST /simulate-leave`: full prediction payload plus `leave_start` and `leave_end`
- `POST /safe-leave-dates`: full prediction payload
- `GET /model-info`: actual training size, evaluation metrics, feature names, and feature importance

Open `http://localhost:5000/` for the testing dashboard. Use the editable requirement field or the 75%, 80%, 85%, and 90% quick buttons. The comparison panel sends the same student scenario at all four requirements and calculates each safe-leave limit through the API.

`as_of_date` is optional and defaults to the server date. It makes tests and historical calculations reproducible.

## Example request

```json
{
  "minimum_attendance": 75,
  "academic_calendar": {"start_date": "2026-06-01", "end_date": "2026-10-31"},
  "holidays": ["2026-08-15"],
  "special_holidays": ["2026-09-21"],
  "working_weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
  "timetable": {
    "Monday": ["Maths", "Maths", "DBMS"],
    "Tuesday": ["AI", "Maths"],
    "Wednesday": ["OS", "Maths"],
    "Thursday": ["DBMS", "AI"],
    "Friday": ["OS", "DBMS"]
  },
  "subjects": {
    "Maths": {"present": 35, "absent": 3},
    "DBMS": {"present": 31, "absent": 5},
    "AI": {"present": 28, "absent": 4},
    "OS": {"present": 34, "absent": 2}
  }
}
```

Example curl on Windows PowerShell:

```powershell
curl.exe -X POST http://localhost:5000/predict-leave -H "Content-Type: application/json" -d "{\"minimum_attendance\":75,\"academic_calendar\":{\"start_date\":\"2026-06-01\",\"end_date\":\"2026-10-31\"},\"working_weekdays\":[\"Monday\",\"Tuesday\",\"Wednesday\",\"Thursday\",\"Friday\"],\"timetable\":{\"Monday\":[\"Maths\"]},\"subjects\":{\"Maths\":{\"present\":35,\"absent\":3}}}"
```

## Response shape

`/predict-leave` returns `status`, a `summary`, subject records containing current and projected class information, `safe_leave_dates`, `warnings`, and `overall_risk`. Percentages are rounded to two decimal places only in the response.

## Leave calculation

For present `P`, absent `A`, and threshold `m`, an additional absence `L` is safe when:

`P / (P + A + L) >= m / 100`

The implementation calculates the largest integer `L` satisfying that inequality and caps it at the number of remaining classes for that subject. Future dates simulate every scheduled class on that date, including duplicate timetable entries.

## ML model

The generator creates realistic semester stages, attendance profiles, configurable requirements from 65% to 90%, class frequency, remaining classes, planned leave, and edge cases. Labels are calculated from projected attendance and relative margins:

- `HIGH`: projected attendance is below the selected minimum
- `MEDIUM`: projected attendance is from the minimum through less than five points above it
- `LOW`: projected attendance is at least five points above it

Features are `current_attendance`, `present_classes`, `absent_classes`, `total_classes`, `remaining_classes`, `weekly_subject_classes`, `minimum_attendance`, `planned_leave_classes`, `days_remaining`, `absence_rate`, and `future_attendance_after_leave`. The target `leave_risk` is never included as a feature. `future_attendance_after_leave` is computed from the proposed leave scenario before prediction, so it is available at inference time.

`train_model.py` fits a 300-tree `RandomForestClassifier` with a fixed seed, uses an 80/20 stratified split, prints accuracy, precision, recall, F1, confusion matrix, and classification report, then saves the model and actual metrics/feature importance to `models/`. Add real rows to `data/real_training_data.csv` using its documented columns and rerun the two training commands to combine them with synthetic data.

The model provides supporting risk analysis and probabilities. Mathematical attendance calculations remain authoritative: a date or leave range is unsafe whenever any projected subject is below the selected minimum, regardless of the model prediction.

## Tests

```bash
pytest -q
```

The tests cover attendance math, holidays, configurable weekdays, repeated classes, future class counts, safe dates, consecutive leave, validation, and Flask endpoints. The API is deliberately JSON-first so a later React, mobile, MongoDB, or Firebase integration can replace storage without changing the calculation contract.
