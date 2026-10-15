import os
import io
import csv
import warnings
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
from sklearn.exceptions import InconsistentVersionWarning

# Suppress sklearn unpickling version warnings
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import joblib

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'student_performance_model.pkl')
FEATURE_NAMES = ['attendance', 'study_hours', 'assignment_score', 'internal_marks', 'previous_marks']

FEATURE_LABELS = {
    'attendance': 'Attendance Rate (%)',
    'study_hours': 'Daily Study Hours (hrs/day)',
    'assignment_score': 'Assignment Score (0-100)',
    'internal_marks': 'Internal Marks (0-100)',
    'previous_marks': 'Previous Exam Marks (0-100)'
}

FEATURE_RANGES = {
    'attendance': {'min': 0.0, 'max': 100.0, 'default': 85.0, 'step': 0.5, 'unit': '%'},
    'study_hours': {'min': 0.0, 'max': 16.0, 'default': 5.0, 'step': 0.1, 'unit': 'hrs/day'},
    'assignment_score': {'min': 0.0, 'max': 100.0, 'default': 80.0, 'step': 0.5, 'unit': '/100'},
    'internal_marks': {'min': 0.0, 'max': 100.0, 'default': 75.0, 'step': 0.5, 'unit': '/100'},
    'previous_marks': {'min': 0.0, 'max': 100.0, 'default': 78.0, 'step': 0.5, 'unit': '/100'}
}

# Load Model
model = None
feature_importances_dict = {}

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
            for name, imp in zip(model.feature_names_in_, model.feature_importances_):
                feature_importances_dict[name] = round(float(imp), 4)
        print(f"Model successfully loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")


def predict_student_performance(attendance, study_hours, assignment_score, internal_marks, previous_marks):
    """
    Predicts final marks using the loaded Random Forest model.
    Matches the user's reference signature.
    """
    df = pd.DataFrame([{
        'attendance': float(attendance),
        'study_hours': float(study_hours),
        'assignment_score': float(assignment_score),
        'internal_marks': float(internal_marks),
        'previous_marks': float(previous_marks)
    }])
    pred = model.predict(df)[0]
    return float(pred)


def get_performance_category(predicted_marks):
    """
    Returns student performance category based on predicted marks.
    """
    marks = float(predicted_marks)
    if marks >= 85:
        return "High Performance (Distinction)"
    elif marks >= 70:
        return "Good Performance (Above Average)"
    elif marks >= 55:
        return "Average Performance (Moderate)"
    elif marks >= 45:
        return "Low Performance (Needs Improvement)"
    else:
        return "Critical Risk (Academic Warning)"


def calculate_grade_and_status(score):
    score = round(float(score), 2)
    category = get_performance_category(score)
    if score >= 90:
        return 'A+', category, 4.0, 'success', 'Outstanding academic mastery with exceptional consistency across all metrics.'
    elif score >= 80:
        return 'A', category, 3.7, 'success', 'Strong comprehension and performance; on track for honors.'
    elif score >= 70:
        return 'B', category, 3.0, 'info', 'Good overall standing with steady performance across core assessments.'
    elif score >= 60:
        return 'C', category, 2.2, 'warning', 'Passing performance; targeted improvement in weaker areas will secure better grades.'
    elif score >= 50:
        return 'D', category, 1.5, 'warning', 'Borderline standing with noticeable vulnerabilities in key evaluated subjects.'
    else:
        return 'F', category, 0.0, 'danger', 'High risk of course failure. Immediate academic mentoring and intervention required.'


def generate_recommendations(features, predicted_score):
    recs = []
    att = float(features.get('attendance', 0))
    study = float(features.get('study_hours', 0))
    assign = float(features.get('assignment_score', 0))
    internal = float(features.get('internal_marks', 0))
    prev = float(features.get('previous_marks', 0))

    # Internal marks impact (Highest importance: ~44.9%)
    if internal < 70:
        recs.append({
            'priority': 'Critical' if internal < 55 else 'High',
            'category': 'Internal Assessments',
            'title': 'Boost Internal Marks (44.9% Model Weight)',
            'description': f'Internal marks carry the highest weight in the model. Improving from {internal:.1f} to 80+ can raise your final score by 6–12 marks.',
            'action': 'Prioritize mid-term preparation, lab evaluations, and continuous unit quizzes.'
        })
    elif internal >= 85:
        recs.append({
            'priority': 'Positive',
            'category': 'Internal Assessments',
            'title': 'Outstanding Internal Standing',
            'description': f'Your internal marks ({internal:.1f}) strongly anchor your high performance prediction.',
            'action': 'Maintain your current study rhythm leading into final exams.'
        })

    # Assignment score impact (28.7% importance)
    if assign < 70:
        recs.append({
            'priority': 'High',
            'category': 'Assignments & Coursework',
            'title': 'Maximize Assignment Submissions',
            'description': f'Assignments constitute 28.7% of the model importance. Current score is {assign:.1f}.',
            'action': 'Submit all problem sets on time, seek TA feedback before deadlines, and review rubrics carefully.'
        })

    # Study hours impact (12.8% importance)
    if study < 3.5:
        recs.append({
            'priority': 'Medium',
            'category': 'Study Habits',
            'title': 'Increase Daily Study Dedication',
            'description': f'Current study duration is {study:.1f} hrs/day. Average high performers average 5.5 to 7.0 hrs/day.',
            'action': 'Add 1.5 to 2 structured hours using Pomodoro intervals (e.g. 50m study / 10m break).'
        })
    elif study >= 7:
        recs.append({
            'priority': 'Positive',
            'category': 'Study Habits',
            'title': 'Commendable Study Discipline',
            'description': f'{study:.1f} hours/day provides deep retention. Ensure quality rest and spaced repetition.',
            'action': 'Balance intense study sessions with adequate sleep and revision.'
        })

    # Attendance impact (6.0% importance & institutional threshold)
    if att < 75:
        recs.append({
            'priority': 'Urgent',
            'category': 'Attendance Warning',
            'title': 'Attendance Below Mandatory 75%',
            'description': f'Current attendance is {att:.1f}%. Falling below 75% incurs institutional penalties and risk of exam debarment.',
            'action': 'Ensure 100% presence in remaining sessions to bring aggregate attendance above 75% immediately.'
        })
    else:
        recs.append({
            'priority': 'Low',
            'category': 'Attendance',
            'title': 'Healthy Attendance Record',
            'description': f'Attendance at {att:.1f}% satisfies all institutional compliance requirements.',
            'action': 'Continue regular lecture engagement and active classroom participation.'
        })

    # Previous marks impact (7.5% importance)
    if prev < 60 and predicted_score > prev:
        recs.append({
            'priority': 'Positive',
            'category': 'Growth Trajectory',
            'title': 'Positive Upward Momentum',
            'description': f'Predicted score ({predicted_score:.1f}) is outperforming previous marks ({prev:.1f}) by +{predicted_score - prev:.1f} points.',
            'action': 'Your effort is paying off—keep the consistent momentum!'
        })

    return recs


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': type(model).__name__ if model else None,
        'features': FEATURE_NAMES
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    return jsonify({
        'model_name': 'Student Performance Predictor',
        'algorithm': 'Random Forest Regressor (Ensemble of 100 Decision Trees)',
        'framework': 'scikit-learn',
        'n_features': len(FEATURE_NAMES),
        'features': FEATURE_NAMES,
        'feature_labels': FEATURE_LABELS,
        'feature_ranges': FEATURE_RANGES,
        'feature_importances': feature_importances_dict,
        'parameters': {
            'n_estimators': getattr(model, 'n_estimators', 100),
            'criterion': getattr(model, 'criterion', 'squared_error'),
            'max_depth': getattr(model, 'max_depth', None) or 'None (unrestricted)',
            'min_samples_split': getattr(model, 'min_samples_split', 2)
        },
        'presets': [
            {
                'name': '🌟 High Achiever',
                'description': 'Top 5% student with consistent high study hours & internal scores',
                'values': {'attendance': 95.0, 'study_hours': 7.2, 'assignment_score': 94.0, 'internal_marks': 92.0, 'previous_marks': 90.0}
            },
            {
                'name': '⚖️ Average Student',
                'description': 'Typical undergraduate student performing in middle 50th percentile',
                'values': {'attendance': 80.0, 'study_hours': 4.5, 'assignment_score': 72.0, 'internal_marks': 68.0, 'previous_marks': 70.0}
            },
            {
                'name': '⚠️ At-Risk Student',
                'description': 'Low attendance and internal marks indicating potential course failure',
                'values': {'attendance': 62.0, 'study_hours': 1.8, 'assignment_score': 45.0, 'internal_marks': 42.0, 'previous_marks': 48.0}
            },
            {
                'name': '🚀 Hard Worker Improver',
                'description': 'High study hours and attendance working to recover from lower past grades',
                'values': {'attendance': 90.0, 'study_hours': 7.5, 'assignment_score': 78.0, 'internal_marks': 65.0, 'previous_marks': 52.0}
            }
        ]
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded on server'}), 500

    data = request.get_json(silent=True) or {}
    try:
        inputs = {}
        for feat in FEATURE_NAMES:
            if feat not in data:
                return jsonify({'error': f'Missing required input feature: {feat}'}), 400
            inputs[feat] = float(data[feat])

        # Prepare DataFrame
        df = pd.DataFrame([inputs])

        # Predict main score via predict_student_performance
        raw_pred = predict_student_performance(
            inputs['attendance'],
            inputs['study_hours'],
            inputs['assignment_score'],
            inputs['internal_marks'],
            inputs['previous_marks']
        )
        score = max(0.0, min(100.0, raw_pred))
        category = get_performance_category(score)

        # Predict across individual trees to compute confidence interval
        tree_predictions = [float(estimator.predict(df)[0]) for estimator in model.estimators_]
        std_dev = float(np.std(tree_predictions))
        ci_lower = max(0.0, score - 1.96 * std_dev)
        ci_upper = min(100.0, score + 1.96 * std_dev)

        grade, status, gpa, badge_color, feedback = calculate_grade_and_status(score)
        recommendations = generate_recommendations(inputs, score)

        # Compute marginal impact for +10% improvement in each metric
        marginal_impacts = {}
        for feat in FEATURE_NAMES:
            modified_inputs = inputs.copy()
            if feat == 'study_hours':
                modified_inputs[feat] = min(16.0, modified_inputs[feat] + 1.5)
            else:
                modified_inputs[feat] = min(100.0, modified_inputs[feat] + 10.0)
            m_df = pd.DataFrame([modified_inputs])
            m_pred = float(model.predict(m_df)[0])
            gain = round(m_pred - score, 2)
            marginal_impacts[feat] = max(0.0, gain)

        return jsonify({
            'success': True,
            'prediction': {
                'score': round(score, 2),
                'predicted_marks': round(score, 2),
                'category': category,
                'performance_category': category,
                'grade': grade,
                'status': status,
                'gpa': gpa,
                'badge_color': badge_color,
                'feedback': feedback,
                'uncertainty_std': round(std_dev, 2),
                'confidence_interval_95': [round(ci_lower, 2), round(ci_upper, 2)],
                'tree_min': round(float(np.min(tree_predictions)), 2),
                'tree_max': round(float(np.max(tree_predictions)), 2)
            },
            'inputs': inputs,
            'marginal_impacts': marginal_impacts,
            'feature_importances': feature_importances_dict,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': f'Prediction computation failed: {str(e)}'}), 500


@app.route('/api/what-if', methods=['POST'])
def what_if():
    """Calculates sensitivity curve for a specific feature while keeping others fixed."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json(silent=True) or {}
    try:
        base_inputs = data.get('base_inputs', {})
        feature_to_vary = data.get('feature', 'study_hours')
        steps = int(data.get('steps', 15))

        if feature_to_vary not in FEATURE_NAMES:
            return jsonify({'error': f'Invalid feature: {feature_to_vary}'}), 400

        # Fill defaults
        inputs = {}
        for f in FEATURE_NAMES:
            inputs[f] = float(base_inputs.get(f, FEATURE_RANGES[f]['default']))

        rng = FEATURE_RANGES[feature_to_vary]
        val_min = float(rng['min'])
        val_max = float(rng['max'])
        values = np.linspace(val_min, val_max, steps)

        rows = []
        for v in values:
            row = inputs.copy()
            row[feature_to_vary] = float(v)
            rows.append(row)

        df = pd.DataFrame(rows)
        preds = model.predict(df)

        curve = []
        for v, p in zip(values, preds):
            curve.append({
                'value': round(float(v), 2),
                'predicted_score': round(float(p), 2)
            })

        return jsonify({
            'success': True,
            'feature': feature_to_vary,
            'feature_label': FEATURE_LABELS.get(feature_to_vary, feature_to_vary),
            'curve': curve
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        rows = []
        # Check if file uploaded
        if 'file' in request.files:
            file = request.files['file']
            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'Please upload a CSV file (.csv)'}), 400
            
            stream = io.StringIO(file.stream.read().decode("utf-8-sig"), newline=None)
            reader = csv.DictReader(stream)
            for r in reader:
                rows.append(r)
        else:
            payload = request.get_json(silent=True)
            if payload and 'rows' in payload:
                rows = payload['rows']

        if not rows:
            return jsonify({'error': 'No data rows found in upload'}), 400

        processed_students = []
        for idx, row in enumerate(rows):
            student_id = row.get('student_id', f'STU-{1000 + idx + 1}')
            name = row.get('name', f'Student {idx + 1}')

            try:
                item = {
                    'attendance': float(row.get('attendance', 75.0)),
                    'study_hours': float(row.get('study_hours', 4.0)),
                    'assignment_score': float(row.get('assignment_score', 70.0)),
                    'internal_marks': float(row.get('internal_marks', 65.0)),
                    'previous_marks': float(row.get('previous_marks', 65.0))
                }
            except (ValueError, TypeError):
                continue

            processed_students.append({
                'student_id': student_id,
                'name': name,
                **item
            })

        if not processed_students:
            return jsonify({'error': 'Could not parse any valid numeric rows from the CSV'}), 400

        df = pd.DataFrame([{k: s[k] for k in FEATURE_NAMES} for s in processed_students])
        predictions = model.predict(df)

        results = []
        scores = []
        grades_count = {'A+': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        risk_count = 0

        for s, pred in zip(processed_students, predictions):
            score = max(0.0, min(100.0, round(float(pred), 2)))
            scores.append(score)
            grade, status, gpa, badge_color, feedback = calculate_grade_and_status(score)
            grades_count[grade] = grades_count.get(grade, 0) + 1
            if score < 60:
                risk_count += 1

            results.append({
                'student_id': s['student_id'],
                'name': s['name'],
                'attendance': s['attendance'],
                'study_hours': s['study_hours'],
                'assignment_score': s['assignment_score'],
                'internal_marks': s['internal_marks'],
                'previous_marks': s['previous_marks'],
                'predicted_score': score,
                'grade': grade,
                'status': status,
                'gpa': gpa,
                'badge_color': badge_color
            })

        summary = {
            'total_students': len(results),
            'average_score': round(float(np.mean(scores)), 2),
            'highest_score': round(float(np.max(scores)), 2),
            'lowest_score': round(float(np.min(scores)), 2),
            'pass_rate': round(float(np.mean([1 if s >= 50 else 0 for s in scores]) * 100), 1),
            'at_risk_count': risk_count,
            'distinction_count': grades_count.get('A+', 0) + grades_count.get('A', 0),
            'grade_distribution': grades_count
        }

        return jsonify({
            'success': True,
            'summary': summary,
            'students': results
        })

    except Exception as e:
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500


@app.route('/api/sample-csv', methods=['GET'])
def download_sample_csv():
    sample_file = os.path.join(app.static_folder, 'sample_students.csv')
    return send_file(sample_file, as_attachment=True, download_name='sample_students.csv', mimetype='text/csv')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" EduPredict AI - Student Performance Prediction Suite")
    print(f" Running at: http://localhost:{port}")
    print(f" Model Loaded: {'YES' if model else 'NO'}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
