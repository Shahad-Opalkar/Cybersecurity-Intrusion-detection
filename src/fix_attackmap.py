# src/fix_attack_map_raw.py
import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
ATTACK_MAP_FILE = os.path.join(MODEL_DIR, "attack_map.joblib")

# KDD columns
KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

print("[INFO] Loading raw training file (no transformations)...")
df = pd.read_csv(TRAIN_FILE, names=KDD_COLUMNS, header=None)

# Ensure we have textual labels; if numeric, the user dataset is encoded
labels = df["label"].astype(str).str.strip().str.lower().values

le = LabelEncoder()
le.fit(labels)
attack_map = {int(i): cls for i, cls in enumerate(le.classes_)}

print("[INFO] attack_map (id -> name) to be saved:")
print(attack_map)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(attack_map, ATTACK_MAP_FILE)
print(f"[INFO] attack_map saved to {ATTACK_MAP_FILE}")
