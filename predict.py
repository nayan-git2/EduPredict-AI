import os
import warnings
import pandas as pd
import joblib
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version warnings
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'student_performance_model.pkl')
if not os.path.exists(MODEL_PATH):
    # Fallback to Downloads folder if running stand-alone
    MODEL_PATH = r"c:\Users\Admin\Downloads\student_performance_model (1).pkl"

model = joblib.load(MODEL_PATH)


def predict_student_performance(attendance, study_hours, assignment_score, internal_marks, previous_marks):
    """
    Predicts student performance using the trained Random Forest model.
    """
    input_data = pd.DataFrame([{
        'attendance': float(attendance),
        'study_hours': float(study_hours),
        'assignment_score': float(assignment_score),
        'internal_marks': float(internal_marks),
        'previous_marks': float(previous_marks)
    }])
    prediction = model.predict(input_data)
    return float(prediction[0])


def get_performance_category(predicted_marks):
    """
    Categorizes the student's performance based on predicted marks.
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


if __name__ == "__main__":
    print("=" * 40)
    print("       Enter Student Details")
    print("=" * 40)

    try:
        attendance = float(
            input("Attendance percentage: ")
        )

        study_hours = float(
            input("Study hours per day: ")
        )

        assignment_score = float(
            input("Assignment score: ")
        )

        internal_marks = float(
            input("Internal marks: ")
        )

        previous_marks = float(
            input("Previous exam marks: ")
        )

        predicted_marks = predict_student_performance(
            attendance,
            study_hours,
            assignment_score,
            internal_marks,
            previous_marks
        )

        category = get_performance_category(
            predicted_marks
        )

        print("\n" + "=" * 40)
        print("         PREDICTION RESULT")
        print("=" * 40)

        print(
            f"Predicted Final Marks: {predicted_marks:.2f}"
        )

        print(
            f"Performance Category: {category}"
        )

        print("=" * 40)

    except ValueError:
        print("\n[Error] Please enter valid numeric values for all fields.")
    except Exception as e:
        print(f"\n[Error] Prediction failed: {e}")
