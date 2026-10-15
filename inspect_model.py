import pickle
import pprint
import sys

path = r"c:\Users\Admin\Downloads\student_performance_model (1).pkl"

try:
    with open(path, 'rb') as f:
        obj = pickle.load(f)

    print("=== MODEL TYPE ===")
    print(type(obj))
    print("\n=== REPR ===")
    print(repr(obj))

    attrs = {}
    for attr in ['feature_names_in_', 'n_features_in_', 'classes_', 'named_steps', 'steps', 'feature_importances_', 'coef_', 'intercept_', 'target_names_']:
        if hasattr(obj, attr):
            attrs[attr] = getattr(obj, attr)

    print("\n=== KEY ATTRIBUTES ===")
    pprint.pprint(attrs)

    if hasattr(obj, 'named_steps'):
        for name, step in obj.named_steps.items():
            print(f"\n--- Step: {name} ({type(step)}) ---")
            if hasattr(step, 'transformers_'):
                for tname, trans, cols in step.transformers_:
                    print(f"  transformer: {tname}, cols: {cols}")
                    if hasattr(trans, 'categories_'):
                        print(f"    categories: {trans.categories_}")
                    if hasattr(trans, 'get_feature_names_out'):
                        try:
                            print(f"    names: {trans.get_feature_names_out()}")
                        except Exception as e:
                            print(f"    error getting names: {e}")

except Exception as ex:
    print(f"Error loading model: {ex}")
    import traceback
    traceback.print_exc()
