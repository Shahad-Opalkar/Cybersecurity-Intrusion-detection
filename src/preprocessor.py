# src/preprocessor.py
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

NUMERIC = [
    "duration","src_bytes","dst_bytes","wrong_fragment","urgent","hot",
    "num_failed_logins","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

CATEGORICAL = ["protocol_type","service","flag","land","logged_in","is_host_login","is_guest_login"]

def load_nslkdd(path):
    """Load NSL-KDD file and coerce numeric columns, keep label as-is."""
    from src.utils.dataset import load_nslkdd as loader
    df = loader(path)
    print("Columns in dataset:", df.columns.tolist())  # 👈 Add this line
    # normalize label types but keep original content
    # coerce numeric columns to numeric
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC),
        ("cat", categorical_transformer, CATEGORICAL)
    ])
    return preprocessor

def save_preprocessor(preprocessor, attack_map):
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    joblib.dump(attack_map, os.path.join(MODEL_DIR, "attack_map.joblib"))
