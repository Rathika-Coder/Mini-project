from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

rng = np.random.default_rng(42)
n = 300

hours_studied   = rng.uniform(0.5, 12, n)
sleep_hours     = rng.uniform(3, 10, n)
revision_count  = rng.uniform(0, 15, n)
stress_level    = rng.uniform(1, 10, n)
subject_raw     = rng.choice([0, 1, 2], n)     # 0=hard, 1=moderate, 2=easy

score = (
    6.5  * hours_studied
    + 2.8  * sleep_hours
    + 2.2  * revision_count
    - 3.5  * stress_level
    + 5.0  * subject_raw
    + rng.normal(0, 4, n)
    + 10
)
score = np.clip(score, 0, 100)

subject_hard     = (subject_raw == 0).astype(float)
subject_moderate = (subject_raw == 1).astype(float)
subject_easy     = (subject_raw == 2).astype(float)

X = np.column_stack([
    hours_studied,
    sleep_hours,
    revision_count,
    stress_level,
    subject_hard,
    subject_moderate,
    subject_easy,
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, score)


def predict(hours, sleep, revisions, stress, subject):
    subject_map = {"hard": [1, 0, 0], "moderate": [0, 1, 0], "easy": [0, 0, 1]}
    sub = subject_map.get(subject, [0, 1, 0])
    x = np.array([[hours, sleep, revisions, stress] + sub], dtype=float)
    x_scaled = scaler.transform(x)
    raw = model.predict(x_scaled)[0]
    return round(float(np.clip(raw, 0, 100)), 1)


def score_grade(score):
    if score >= 90:
        return "A+", "Excellent"
    elif score >= 80:
        return "A", "Great"
    elif score >= 70:
        return "B", "Good"
    elif score >= 60:
        return "C", "Average"
    elif score >= 50:
        return "D", "Below Average"
    else:
        return "F", "Needs Improvement"


@app.route("/", methods=["GET", "POST"])
def index():
    predicted_score = None
    grade = None
    grade_label = None
    form_data = {}
    error = None

    if request.method == "POST":
        try:
            hours     = float(request.form.get("hours", 0))
            sleep     = float(request.form.get("sleep", 0))
            revisions = float(request.form.get("revisions", 0))
            stress    = float(request.form.get("stress", 5))
            subject   = request.form.get("subject", "moderate")

            form_data = {
                "hours": hours,
                "sleep": sleep,
                "revisions": revisions,
                "stress": int(stress),
                "subject": subject,
            }

            errors = []
            if not (0 <= hours <= 24):
                errors.append("Study hours must be between 0 and 24.")
            if not (0 <= sleep <= 24):
                errors.append("Sleep hours must be between 0 and 24.")
            if not (0 <= revisions <= 50):
                errors.append("Revision sessions must be between 0 and 50.")
            if not (1 <= stress <= 10):
                errors.append("Stress level must be between 1 and 10.")
            if subject not in ("hard", "moderate", "easy"):
                errors.append("Invalid subject difficulty.")

            if errors:
                error = " ".join(errors)
            else:
                predicted_score = predict(hours, sleep, revisions, stress, subject)
                grade, grade_label = score_grade(predicted_score)
        except (ValueError, TypeError):
            error = "Please enter valid numeric values for all fields."

    return render_template(
        "index.html",
        predicted_score=predicted_score,
        grade=grade,
        grade_label=grade_label,
        form_data=form_data,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
