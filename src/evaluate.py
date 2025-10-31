import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.utils.dataset import load_nslkdd

# -------------------------------------------------------------------
# 1️⃣ Setup and Path Definitions
# -------------------------------------------------------------------
print("[INFO] Loading model, preprocessor, and label encoder...")

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../Project1/src
PROJECT_DIR = os.path.dirname(BASE_DIR)                     # .../Project1
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# -------------------------------------------------------------------
# 2️⃣ Detect Available Model
# -------------------------------------------------------------------
if os.path.exists(os.path.join(MODEL_DIR, "multiclass_lgbm.joblib")):
    MODEL_PATH = os.path.join(MODEL_DIR, "multiclass_lgbm.joblib")
    MODEL_TYPE = "Multiclass"
elif os.path.exists(os.path.join(MODEL_DIR, "binary_lgbm.joblib")):
    MODEL_PATH = os.path.join(MODEL_DIR, "binary_lgbm.joblib")
    MODEL_TYPE = "Binary"
else:
    raise FileNotFoundError("❌ No trained LightGBM model found in 'models/'")

ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"❌ Label encoder not found at: {ENCODER_PATH}")

# -------------------------------------------------------------------
# 3️⃣ Load Model, Encoder, and Optional Preprocessor
# -------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

if os.path.exists(PREPROCESSOR_PATH):
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("✅ Preprocessor loaded successfully.")
else:
    preprocessor = None
    print("⚠️ No preprocessor found — proceeding without it.")

print(f"✅ Loaded {MODEL_TYPE} model from: {MODEL_PATH}")

# -------------------------------------------------------------------
# 4️⃣ Load NSL-KDD Test Dataset
# -------------------------------------------------------------------
print("[INFO] Loading NSL-KDD Test Data...")
DATA_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Test dataset not found at: {DATA_PATH}")

result = load_nslkdd(DATA_PATH)

# Handle different return types
if isinstance(result, pd.DataFrame):
    df = result.copy()
    if 'label' in df.columns:
        y_true = df['label']
        X_test = df.drop(columns=['label'])
    else:
        raise ValueError("❌ 'label' column not found in dataset.")
elif isinstance(result, tuple):
    if len(result) == 2:
        X_test, y_true = result
    elif len(result) == 3:
        X_test, y_true, _ = result
    else:
        raise ValueError("Unexpected number of return values from load_nslkdd()")
else:
    raise TypeError("load_nslkdd() returned unsupported type.")

print(f"✅ Loaded dataset: {DATA_PATH} | Shape: {X_test.shape}")

# -------------------------------------------------------------------
# 5️⃣ Preprocess Test Data and Predict
# -------------------------------------------------------------------
print("[INFO] Preprocessing test data and predicting...")
X_test_proc = preprocessor.transform(X_test) if preprocessor else X_test
y_pred = model.predict(X_test_proc)

# -------------------------------------------------------------------
# 6️⃣ Decode Labels
# -------------------------------------------------------------------
y_pred_decoded = label_encoder.inverse_transform(y_pred)
y_true_decoded = label_encoder.inverse_transform(y_true)

# -------------------------------------------------------------------
# 7️⃣ Evaluation Metrics
# -------------------------------------------------------------------
print("\n📊 Model Evaluation Results:")
print("------------------------------------------------")
print(classification_report(y_true_decoded, y_pred_decoded))
print("------------------------------------------------")
print("Confusion Matrix:")
cm = confusion_matrix(y_true_decoded, y_pred_decoded)
print(cm)
print("------------------------------------------------")
print(f"✅ Accuracy: {accuracy_score(y_true_decoded, y_pred_decoded) * 100:.2f}%")

# -------------------------------------------------------------------
# 8️⃣ Visualize and Save Confusion Matrix
# -------------------------------------------------------------------
print("[INFO] Generating confusion matrix visualization...")

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)

plt.title(f"Confusion Matrix ({MODEL_TYPE} LGBM Model on NSL-KDD Test Set)", fontsize=14)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.tight_layout()

# Save plot
os.makedirs(RESULTS_DIR, exist_ok=True)
SAVE_PATH = os.path.join(RESULTS_DIR, f"confusion_matrix_{MODEL_TYPE.lower()}.png")
plt.savefig(SAVE_PATH, dpi=300)

print(f"✅ Confusion matrix saved to: {SAVE_PATH}")
plt.show()
