# src/check_predictions.py
import pandas as pd
import joblib
import os
from collections import Counter
from src.preprocessor import load_nslkdd, NUMERIC, CATEGORICAL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

PREPROCESSOR_FILE = os.path.join(MODELS_DIR, "preprocessor.joblib")
MULTI_MODEL_FILE = os.path.join(MODELS_DIR, "multiclass_lgbm.joblib")
ATTACK_MAP_FILE = os.path.join(MODELS_DIR, "attack_map.joblib")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")

print("[INFO] Loading artifacts...")
preprocessor = joblib.load(PREPROCESSOR_FILE)
model = joblib.load(MULTI_MODEL_FILE)
attack_map = joblib.load(ATTACK_MAP_FILE)

print("[INFO] Loading test data...")
df = load_nslkdd(TEST_FILE)
X = df[NUMERIC + CATEGORICAL]
y_raw = df["label"].astype(str).values

print("[INFO] Transforming and predicting...")
X_t = preprocessor.transform(X)
preds = model.predict(X_t)

# Map preds to names using attack_map (id -> name)
pred_names = [attack_map.get(int(p), str(p)) for p in preds]

print("\nPREDICTION DISTRIBUTION (names):")
print(Counter(pred_names))

# save sample compare
sample_df = pd.DataFrame({"Actual_raw": y_raw, "Predicted_name": pred_names})
sample_df.to_csv(os.path.join(MODELS_DIR, "prediction_sample.csv"), index=False)
print("[INFO] Saved prediction_sample.csv")
