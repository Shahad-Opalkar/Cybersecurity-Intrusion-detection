# src/train_models.py
import os
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from src.preprocessor import load_nslkdd, NUMERIC, CATEGORICAL, build_preprocessor
from collections import Counter

# =============================
# 📁 Paths and setup
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")

BINARY_MODEL_FILE = os.path.join(MODEL_DIR, "binary_lgbm.joblib")
MULTICLASS_MODEL_FILE = os.path.join(MODEL_DIR, "multiclass_lgbm.joblib")
PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "preprocessor.joblib")
ATTACK_MAP_FILE = os.path.join(MODEL_DIR, "attack_map.joblib")
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.joblib")

RANDOM_STATE = 42

# =============================
# 🧾 Load datasets
# =============================
print("📂 Loading datasets...")
train_df = load_nslkdd(TRAIN_FILE)
test_df = load_nslkdd(TEST_FILE)

print(f"✅ Train shape: {train_df.shape}")
print(f"✅ Test shape: {test_df.shape}")

# =============================
# 🏷️ Prepare labels
# =============================
y_train_raw = train_df["label"].astype(str).str.strip().str.lower().values
y_test_raw = test_df["label"].astype(str).str.strip().str.lower().values

# Combine both label sets to include unseen labels (like 'saint')
all_labels = np.unique(np.concatenate([y_train_raw, y_test_raw, ["unknown"]]))

le = LabelEncoder()
le.fit(all_labels)

# --- Safe transform function ---
def safe_transform(encoder, labels):
    classes = set(encoder.classes_)
    safe_labels = [lbl if lbl in classes else "unknown" for lbl in labels]
    return encoder.transform(safe_labels)

y_train_enc = safe_transform(le, y_train_raw)
y_test_enc = safe_transform(le, y_test_raw)

# Save the label encoder for use in evaluation
joblib.dump(le, LABEL_ENCODER_FILE)

# Create attack map for readability
attack_map = {int(i): cls for i, cls in enumerate(le.classes_)}
joblib.dump(attack_map, ATTACK_MAP_FILE)
print(f"✅ Label encoder and attack map saved with {len(attack_map)} classes.")

# =============================
# ⚙️ Preprocessing
# =============================
print("⚙️ Building preprocessor...")
X_train = train_df[NUMERIC + CATEGORICAL]
X_test = test_df[NUMERIC + CATEGORICAL]

preprocessor = build_preprocessor()
X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)

joblib.dump(preprocessor, PREPROCESSOR_FILE)
print("✅ Preprocessor saved.")

# =============================
# 🧠 Binary Classifier
# =============================
print("\n🚀 Training Binary Classifier...")

normal_idx = None
for idx, lbl in enumerate(le.classes_):
    if "normal" in lbl:
        normal_idx = idx
        break

if normal_idx is None:
    raise ValueError("❌ 'normal' class not found in labels!")

y_train_bin = (y_train_enc != normal_idx).astype(int)
y_test_bin = (y_test_enc != normal_idx).astype(int)

bin_model = LGBMClassifier(objective="binary", random_state=RANDOM_STATE)
bin_model.fit(X_train_t, y_train_bin)
joblib.dump(bin_model, BINARY_MODEL_FILE)
print("✅ Binary model saved.")

# =============================
# 🧠 Multiclass Classifier
# =============================
print("\n🚀 Applying SMOTE and training Multiclass Classifier...")

# Count samples per class
class_counts = Counter(y_train_enc)
min_class_samples = min(class_counts.values())

# Adjust k_neighbors safely
k_neighbors = max(1, min(5, min_class_samples - 1))
print(f"Using k_neighbors={k_neighbors} for SMOTE (min class size = {min_class_samples})")

# Apply SMOTE with adjusted neighbors
smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
X_res, y_res = smote.fit_resample(X_train_t, y_train_enc)

multi_model = LGBMClassifier(
    objective="multiclass",
    num_class=len(np.unique(y_res)),
    random_state=RANDOM_STATE
)
multi_model.fit(X_res, y_res)
joblib.dump(multi_model, MULTICLASS_MODEL_FILE)
print("✅ Multiclass model saved.")

# =============================
# 📊 Evaluation
# =============================
print("\n📊 Evaluating models...")

y_pred_bin = bin_model.predict(X_test_t)
print("\nBinary Classification Report:")
print(classification_report(y_test_bin, y_pred_bin, zero_division=0))
print("Binary Accuracy:", accuracy_score(y_test_bin, y_pred_bin))

y_pred_multi = multi_model.predict(X_test_t)
pred_names = [attack_map[int(p)] for p in y_pred_multi]
true_names = [attack_map[int(p)] for p in y_test_enc]

print("\nMulticlass Classification Report:")
print(classification_report(true_names, pred_names, zero_division=0))
print("Multiclass Accuracy:", accuracy_score(true_names, pred_names))

print("\n✅ Training complete! Models and encoder saved in 'models/'")
