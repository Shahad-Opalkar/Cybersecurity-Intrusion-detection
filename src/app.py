# src/app.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

from src.preprocessor import NUMERIC, CATEGORICAL
from src.explain import explain_multi, FEATURE_NAMES

# ===============================
# Paths
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "preprocessor.joblib")
BINARY_MODEL_FILE = os.path.join(MODEL_DIR, "binary_lgbm.joblib")
MULTICLASS_MODEL_FILE = os.path.join(MODEL_DIR, "multiclass_lgbm.joblib")
ATTACK_MAP_FILE = os.path.join(MODEL_DIR, "attack_map.joblib")

# ===============================
# Streamlit Config
# ===============================
st.set_page_config(layout="wide", page_title="AI IDS Demo")

if "running" not in st.session_state:
    st.session_state.running = False
if "seen_idx" not in st.session_state:
    st.session_state.seen_idx = set()

# ===============================
# Load Artifacts
# ===============================
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    bin_model = joblib.load(BINARY_MODEL_FILE)
    multi_model = joblib.load(MULTICLASS_MODEL_FILE)
    attack_map = joblib.load(ATTACK_MAP_FILE)
    return preprocessor, bin_model, multi_model, attack_map

try:
    preprocessor, bin_model, multi_model, attack_map = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# ===============================
# UI Layout
# ===============================
st.title("🛡️ AI-powered IDS — Demo")
st.markdown("Binary detection → Multi-class classification → SHAP explanations")

colL, colR = st.columns([2, 1])

with colL:
    st.header("Input / Mode")
    mode = st.radio("Mode", ["Simulated Live Stream", "Upload CSV (batch)"])

    if mode == "Simulated Live Stream":
        stream_file = st.text_input("Stream file path", os.path.join(BASE_DIR, "..", "stream_out.csv"))
        delay = st.slider("Polling interval (seconds)", 0.5, 3.0, 1.0)

        start_btn = st.button("Start Live Monitoring")
        stop_btn = st.button("Stop")

        if start_btn:
            st.session_state.running = True
            st.rerun()
        if stop_btn:
            st.session_state.running = False
            st.rerun()

    else:
        uploaded = st.file_uploader("Upload NSL-KDD formatted CSV (no header)", type=["csv", "txt"])
        if uploaded:
            cols = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "KDDTrain+.txt"), nrows=0).columns
            df_up = pd.read_csv(uploaded, names=cols, header=None)
            st.dataframe(df_up.head())

with colR:
    st.header("Models")
    st.write("Binary:", os.path.basename(BINARY_MODEL_FILE))
    st.write("Multi-class:", os.path.basename(MULTICLASS_MODEL_FILE))
    st.write("Attack map (id → name):")
    st.json(attack_map)

live_area = st.empty()

# ===============================
# Helper: Process One Row
# ===============================
def process_row(row_series):
    req = NUMERIC + CATEGORICAL
    row = row_series[req]
    df_row = pd.DataFrame([row.values], columns=req)

    for c in NUMERIC:
        df_row[c] = pd.to_numeric(df_row[c], errors="coerce")

    X_t = preprocessor.transform(df_row)
    prob_attack = float(bin_model.predict_proba(X_t)[:, 1][0])
    is_attack = int(prob_attack >= 0.5)

    if is_attack:
        pred_idx = int(multi_model.predict(X_t)[0])
        probs = multi_model.predict_proba(X_t)[0]

        # Fix: ensure index is valid
        if pred_idx >= len(probs):
            pred_idx = int(np.argmax(probs))

        conf = float(probs[pred_idx])
        attack_label = attack_map.get(pred_idx, f"Unknown ({pred_idx})")

        # SHAP explanation (safe)
        try:
            _, top_multi = explain_multi(X_t.flatten())
        except Exception:
            top_multi = []

        return {
            "attack": True,
            "prob_attack": prob_attack,
            "attack_label": attack_label,
            "confidence": conf,
            "top_multi": top_multi
        }
    else:
        return {"attack": False, "prob_attack": prob_attack}

# ===============================
# Simulated Streaming
# ===============================
if mode == "Simulated Live Stream" and st.session_state.running:
    if not os.path.exists(stream_file):
        st.warning("Stream file not found. Run stream_simulator.py first.")
    else:
        while st.session_state.running:
            try:
                df_stream = pd.read_csv(stream_file)
            except Exception as e:
                st.error(f"Error reading stream file: {e}")
                break

            for idx, row in df_stream.iterrows():
                if idx in st.session_state.seen_idx:
                    continue

                st.session_state.seen_idx.add(idx)
                res = process_row(row)

                with live_area.container():
                    if res["attack"]:
                        st.error(f"ALERT — {res['attack_label']} (P_attack={res['prob_attack']:.2f})")
                        st.write(f"Confidence: {res['confidence']:.2f}")
                        st.subheader("Top features (multi-class SHAP)")
                        st.table(pd.DataFrame(res["top_multi"], columns=["feature", "shap_value"]))
                    else:
                        st.success(f"No attack detected (P_attack={res['prob_attack']:.2f})")

                time.sleep(delay)

            time.sleep(1)

# ===============================
# Batch Inference Mode
# ===============================
if mode == "Upload CSV (batch)" and "uploaded" in locals() and uploaded:
    if st.button("Run batch inference"):
        df_in = df_up.copy()
        try:
            X_t = preprocessor.transform(df_in[NUMERIC + CATEGORICAL])
        except Exception as e:
            st.error(f"Uploaded CSV columns don't match required columns: {e}")
        else:
            probs = bin_model.predict_proba(X_t)[:, 1]
            preds = (probs >= 0.5).astype(int)
            st.write(pd.Series(preds).value_counts())
