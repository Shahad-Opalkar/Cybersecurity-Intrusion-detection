#  Cybersecurity Intrusion Detection System (IDS) using AI/ML

This project implements a **Network Intrusion Detection System (IDS)** using **Machine Learning** (LightGBM) trained on the **NSL-KDD Dataset**.  
It classifies network traffic into multiple attack categories for enhanced cybersecurity monitoring.

---

##  Features
- Multiclass classification (Normal & Attack types)
- Preprocessing pipeline with categorical + numerical handling
- SMOTE-based oversampling for imbalance correction
- Trained using LightGBM with 71% multiclass accuracy
- Confusion matrix and classification metrics

---

##  Project Structure
Cybersecurity-IDS-Project/
│
├── data/
│ ├── KDDTrain+.txt
│ ├── KDDTest+.txt
│
├── models/
│ ├── multiclass_lgbm.joblib
│ ├── label_encoder.joblib
│ ├── attack_map.joblib
│
├── src/
│ ├── preprocessor.py
│ ├── train_models.py
│ ├── evaluate.py
│ ├── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
