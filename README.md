# 🎓 EduPredict AI — Student Performance Prediction System

A modern, full-stack Machine Learning web application powered by your trained **Random Forest Regressor** model (`student_performance_model.pkl`). 

Designed with modern UI aesthetics (Tailwind CSS, Glassmorphism, Chart.js, Lucide icons), it provides real-time individual predictions, uncertainty estimation, academic advisories, "What-If" sensitivity simulations, and cohort batch analytics.

---

## 🚀 Quick Start

### Option 1: Double-click Launcher (Windows)
Simply double click:
```bash
run.bat
```
This automatically ensures dependencies and launches the server.

### Option 2: Command Line
```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run the application
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser!

---

## 🧠 Machine Learning Model Details

| Attribute | Specification |
| :--- | :--- |
| **Model Algorithm** | `RandomForestRegressor` (Ensemble of 100 Decision Trees) |
| **Target Variable** | Student Final Exam / Course Score (Continuous scale 0–100) |
| **Criterion** | Squared Error (`squared_error`) |
| **Framework** | `scikit-learn` & `joblib` |
| **Input Features** | 5 core academic predictors |

### Feature Importance Weights
The model places data-driven weights across student behaviors:
1. **Internal Marks (`internal_marks`)**: **44.85%** (Single largest driver)
2. **Assignment Score (`assignment_score`)**: **28.74%**
3. **Daily Study Hours (`study_hours`)**: **12.83%**
4. **Previous Exam Marks (`previous_marks`)**: **7.53%**
5. **Attendance Rate (`attendance`)**: **6.04%** (With mandatory 75% institutional compliance cutoff)

---

## ✨ Features & Capabilities

1. **Interactive Individual Prediction Dashboard**:
   - Dual numeric input + synchronized smooth slider controls.
   - Quick Archetype Presets: *High Achiever*, *Average Student*, *At-Risk Student*, *Hard Worker Improver*.
   - Instant calculation mode with SVG radial progress gauge.
   - Grade Classification (`A+`, `A`, `B`, `C`, `D`, `F`), GPA (4.0 scale), and Standing.
   - **Model Uncertainty & 95% Confidence Interval**: Evaluates variance across all 100 decision trees.
   - **Interactive Radar Chart**: Compares student metrics against top honor performers.
   - **Marginal Gains Matrix**: Highlights projected score gains for each incremental effort.
   - **AI Academic Advisory**: Actionable interventions prioritized by model leverage.

2. **What-If Sensitivity Simulator**:
   - Dynamic curve analysis to visualize how changing a feature impacts the final grade while locking others.

3. **Cohort Batch Prediction & Export**:
   - Drag & drop CSV file uploader.
   - Built-in **12-Student Sample** button for instant one-click demonstration.
   - Cohort KPIs: Total Students, Class Average, Pass Rate, Distinction Count, At-Risk Count.
   - Searchable, filterable roster table.
   - One-click CSV export with predicted scores and grades.

4. **Model Transparency & Developer REST API**:
   - Explains tree bagging architecture and Gini importances.
   - Live REST API documentation with `cURL` request snippet and sample JSON output.

---

## 📡 REST API Reference

### Single Prediction
`POST /api/predict`
```json
{
  "attendance": 85.0,
  "study_hours": 5.0,
  "assignment_score": 80.0,
  "internal_marks": 75.0,
  "previous_marks": 78.0
}
```

**Response**:
```json
{
  "success": true,
  "prediction": {
    "score": 75.28,
    "grade": "B",
    "status": "Proficient",
    "gpa": 3.0,
    "uncertainty_std": 1.82,
    "confidence_interval_95": [71.71, 78.85]
  },
  "marginal_impacts": {
    "internal_marks": 4.5,
    "assignment_score": 2.8,
    "study_hours": 1.5,
    "previous_marks": 0.8,
    "attendance": 0.4
  }
}
```

---

## 📂 Project Architecture

```
Ml Project/
├── model/
│   └── student_performance_model.pkl   # The trained scikit-learn model
├── static/
│   ├── css/
│   │   └── style.css                   # Custom styles, animations & glassmorphism
│   ├── js/
│   │   └── app.js                      # Reactive frontend logic, Chart.js & API sync
│   └── sample_students.csv             # 12 pre-configured diverse student records
├── templates/
│   └── index.html                      # Modern responsive Single Page App UI
├── app.py                              # Flask backend with ML prediction APIs
├── requirements.txt                    # Python dependencies
├── run.bat                             # One-click Windows runner
└── README.md
```
