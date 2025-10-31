# src/explain.py
import joblib
import numpy as np
import pandas as pd
import shap
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "preprocessor.joblib")
MULTICLASS_MODEL_FILE = os.path.join(MODEL_DIR, "multiclass_lgbm.joblib")
ATTACK_MAP_FILE = os.path.join(MODEL_DIR, "attack_map.joblib")

# Load artifacts
preprocessor = joblib.load(PREPROCESSOR_FILE)
multi_model = joblib.load(MULTICLASS_MODEL_FILE)
attack_map = joblib.load(ATTACK_MAP_FILE)  # id -> name

# Build FEATURE_NAMES from preprocessor
def get_feature_names(preprocessor):
    num_names = preprocessor.transformers_[0][2]
    cat_step = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = preprocessor.transformers_[1][2]
    try:
        cat_feature_names = list(cat_step.get_feature_names_out(cat_cols))
    except Exception:
        cat_feature_names = []
        for i, c in enumerate(cat_cols):
            cats = cat_step.categories_[i]
            cat_feature_names += [f"{c}__{v}" for v in cats]
    return list(num_names) + cat_feature_names

FEATURE_NAMES = get_feature_names(preprocessor)

# SHAP explainers for tree models
explainer_multi = shap.TreeExplainer(multi_model)
def explain_multi(sample_array):
    """
    sample_array: 1d numpy after preprocessing
    returns: (predicted_idx, [(feature, shap_value)...])
    """
    arr2d = sample_array.reshape(1, -1)
    probs = multi_model.predict_proba(arr2d)[0]
    top_class = int(np.argmax(probs))

    shap_values = explainer_multi.shap_values(arr2d)

    # Handle shap output format (list vs array)
    if isinstance(shap_values, list):
        class_shap = np.array(shap_values[top_class]).flatten()
    else:
        class_shap = np.array(shap_values).flatten()

    # --- Fix: Align lengths ---
    min_len = min(len(FEATURE_NAMES), len(class_shap))
    feature_names = FEATURE_NAMES[:min_len]
    class_shap = class_shap[:min_len]

    df = pd.DataFrame({
        "feature": feature_names,
        "shap": class_shap
    })

    df["abs"] = df["shap"].abs()
    df = df.sort_values("abs", ascending=False).head(8)
    return top_class, df[["feature", "shap"]].values.tolist()
