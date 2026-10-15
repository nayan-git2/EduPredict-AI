import os
import json
import warnings
import joblib
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
warnings.filterwarnings('ignore', category=UserWarning)

model_path = os.path.join(os.path.dirname(__file__), 'model', 'student_performance_model.pkl')
model = joblib.load(model_path)

trees_data = []
for est in model.estimators_:
    t = est.tree_
    trees_data.append({
        'feature': t.feature.tolist(),
        'threshold': [round(float(x), 6) for x in t.threshold],
        'children_left': t.children_left.tolist(),
        'children_right': t.children_right.tolist(),
        'value': [round(float(v[0][0]), 6) for v in t.value]
    })

forest_export = {
    'features': list(model.feature_names_in_),
    'feature_importances': {
        name: round(float(imp), 4)
        for name, imp in zip(model.feature_names_in_, model.feature_importances_)
    },
    'n_estimators': len(trees_data),
    'trees': trees_data
}

output_path = os.path.join(os.path.dirname(__file__), 'static', 'model.json')
with open(output_path, 'w') as f:
    json.dump(forest_export, f)

print(f"Exported model with {len(trees_data)} trees to {output_path} ({os.path.getsize(output_path)} bytes)")
