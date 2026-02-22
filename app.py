import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import bcrypt
import random
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# -------------------------------------------------
# Flask Configuration
# -------------------------------------------------
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # allow React frontend
UPLOAD_FOLDER = "uploads/synthea_data_csv"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {".csv"}

# -------------------------------------------------
# User Database Setup
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password BLOB
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def load_csv(file_name, folder):
    path = os.path.join(UPLOAD_FOLDER, folder, file_name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def convert_dates(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].apply(
                lambda x: x.tz_localize(None)
                if hasattr(x, "tzinfo") and x is not pd.NaT and x.tzinfo
                else x
            )
    return df

def safe_datetime(val):
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    return val

def sanitize_patient(patient):
    if not isinstance(patient, dict):
        return patient
    sanitized = {}
    for k, v in patient.items():
        if isinstance(v, dict):
            sanitized[k] = sanitize_patient(v)
        elif isinstance(v, list):
            sanitized[k] = [
                sanitize_patient(i) if isinstance(i, dict) else safe_datetime(i)
                for i in v
            ]
        else:
            sanitized[k] = safe_datetime(v)
    return sanitized

def is_real_health_condition(desc):
    if pd.isna(desc):
        return False
    desc_lower = desc.lower()
    keywords = [
        "disorder", "disease", "syndrome", "cancer", "diabetes", "asthma", "pain",
        "chronic", "hypertension", "infection", "injury", "stroke", "fracture",
        "renal", "liver", "hepatitis", "arthritis", "migraine", "depression", "anxiety"
    ]
    return any(k in desc_lower for k in keywords)

# -------------------------------------------------
# Patient Preprocessing
# -------------------------------------------------
def preprocess_patient_details(email):
    folder = email
    patients = load_csv("patients.csv", folder)
    if patients.empty:
        return []

    patients = convert_dates(patients, ["BIRTHDATE"])
    datasets = {
        "conditions": load_csv("conditions.csv", folder),
        "medications": load_csv("medications.csv", folder),
        "procedures": load_csv("procedures.csv", folder),
        "observations": load_csv("observations.csv", folder),
    }

    for name, df in datasets.items():
        if name == "observations":
            datasets[name] = convert_dates(df, ["DATE"])
        else:
            datasets[name] = convert_dates(df, ["START", "STOP"])
        datasets[name] = df.drop_duplicates()

    all_patients = []

    for _, row in patients.iterrows():
        pid = str(row.get("Id", "")).strip()
        age = int((pd.Timestamp.now() - row["BIRTHDATE"]).days / 365) if pd.notna(row.get("BIRTHDATE")) else None
        patient_conditions = datasets["conditions"][datasets["conditions"]["PATIENT"] == pid]
        patient_conditions = patient_conditions[patient_conditions["DESCRIPTION"].apply(is_real_health_condition)]
        last_condition = patient_conditions.sort_values("START", ascending=False).iloc[0] if not patient_conditions.empty else None

        def filter_after_date(df, date_col):
            df_patient = df[df["PATIENT"] == pid].copy()
            return df_patient.to_dict(orient="records")

        patient_detail = {
            "id": pid,
            "name": f"{row.get('FIRST','')} {row.get('LAST','')}".strip() or "Unnamed Patient",
            "birthdate": row.get("BIRTHDATE"),
            "age": age,
            "gender": row.get("GENDER", "N/A"),
            "city": row.get("ADDRESS_CITY", "N/A"),
            "state": row.get("ADDRESS_STATE", "N/A"),
            "country": row.get("ADDRESS_COUNTRY", "N/A"),
            "current_condition": last_condition.to_dict() if last_condition is not None else {},
            "conditions": filter_after_date(datasets["conditions"], "START"),
            "medications": filter_after_date(datasets["medications"], "START"),
            "observations": filter_after_date(datasets["observations"], "DATE"),
            "procedures": filter_after_date(datasets["procedures"], "START"),
        }
        all_patients.append(patient_detail)
    return all_patients

# -------------------------------------------------
# Load Models
# -------------------------------------------------
MODEL_DIR = "model_outputs"
risk_model = future_model = treatment_model = None
risk_label_encoder = treatment_label_encoder = None
feature_cols = []

# Current Risk Model
risk_model_path = os.path.join(MODEL_DIR, "current_risk_classifier.pkl")
if os.path.exists(risk_model_path):
    with open(risk_model_path, "rb") as f:
        clf_bundle = pickle.load(f)
        risk_model = clf_bundle["model"]
        risk_label_encoder = clf_bundle.get("label_encoder") or clf_bundle.get("risk_label_encoder")
        feature_cols = clf_bundle["feature_cols"]
        print("✅ Risk model loaded.")

# Future Risk Model
future_model_path = os.path.join(MODEL_DIR, "future_5yr_regressor.pkl")
if os.path.exists(future_model_path):
    with open(future_model_path, "rb") as f:
        bundle = pickle.load(f)
        future_model = bundle.get("model", bundle)
        print("✅ Future risk model loaded.")

# Treatment Model
treat_model_path = os.path.join(MODEL_DIR, "treatment_recommender.pkl")
if os.path.exists(treat_model_path):
    with open(treat_model_path, "rb") as f:
        treatment_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "treatment_label_encoder.pkl"), "rb") as f:
        treatment_label_encoder = pickle.load(f)
    print("✅ Treatment model loaded.")

# -------------------------------------------------
# Risk Helper
# -------------------------------------------------
def risk_label_from_score(score):
    if score >= 0.5:
        return "High"
    elif score >= 0.3:
        return "Medium"
    else:
        return "Low"

def risk_numeric_from_label(label):
    if not label:
        return 0
    l = str(label).lower()
    if l == "high":
        return 3
    elif l == "medium":
        return 2
    elif l == "low":
        return 1
    else:
        return 0

# -------------------------------------------------
# AUTH API - Register
# -------------------------------------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_pw))
        conn.commit()
        conn.close()
        return jsonify({"message": "User registered successfully"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "User already exists"}), 400

# -------------------------------------------------
# AUTH API - Login
# -------------------------------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return jsonify({"error": "User not found"}), 404

        stored_pw = row[0]
        if bcrypt.checkpw(password.encode("utf-8"), stored_pw):
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"error": "Incorrect password"}), 401

    except Exception as e:
        print(f"❌ Error in /login: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# CSV Upload API
# -------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_files():
    email = request.form.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    user_folder = os.path.join(UPLOAD_FOLDER, email)
    os.makedirs(user_folder, exist_ok=True)

    files = request.files.getlist("files")
    saved_files = []

    for file in files:
        filename = os.path.basename(file.filename)
        if allowed_file(filename):
            file.save(os.path.join(user_folder, filename))
            saved_files.append(filename)

    if not saved_files:
        return jsonify({"error": "No valid CSV files uploaded"}), 400

    return jsonify({"message": "Files uploaded successfully", "files": saved_files})

# -------------------------------------------------
# Get Patients Summary
# -------------------------------------------------
@app.route("/patients", methods=["GET"])
def get_patients():
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    all_patients = preprocess_patient_details(email)
    summary = [
        {
            "id": p["id"],
            "name": p["name"],
            "age": p["age"],
            "gender": p["gender"],
            "condition": p["current_condition"].get("DESCRIPTION", "N/A"),
        }
        for p in all_patients
    ]
    return jsonify({"patients": summary})

# -------------------------------------------------
# Patient Details
# -------------------------------------------------
@app.route("/patient/<patient_id>", methods=["GET"])
def get_patient_details(patient_id):
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    all_patients = preprocess_patient_details(email)
    patient = next((p for p in all_patients if str(p["id"]) == str(patient_id)), None)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    return jsonify(sanitize_patient(patient))

# -------------------------------------------------
# Predictive Analysis Summary
# -------------------------------------------------
@app.route("/predictive_analysis", methods=["GET"])
def predictive_analysis():
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400

    try:
        patients = preprocess_patient_details(email)
        if not patients:
            return jsonify({"error": "No patient data found"}), 404

        summary_data = []

        for p in patients:
            cond_desc = str(p.get("current_condition", {}).get("DESCRIPTION", "Unknown")).lower()

            # Simulated features
            features = {
                "AGE": p.get("age", 0),
                "procedure_severity": random.randint(0, 4),
                "medication_intensity": random.randint(0, 3),
                "treatment_response": random.randint(0, 2),
                "observation_value": random.randint(0, 100),
                "age_over_60": 1 if p.get("age", 0) > 60 else 0,
                "procedure_count": len(p.get("procedures", [])),
                "medication_count": len(p.get("medications", [])),
                "careplan_count": len(p.get("conditions", [])),
                "avg_observation_value": random.randint(0, 100)
            }

            predicted_risk = "N/A"
            if risk_model is not None:
                try:
                    X = pd.DataFrame([features])[feature_cols]
                    if hasattr(risk_model, "predict_proba"):
                        probs = risk_model.predict_proba(X)[0]
                        pred = np.argmax(probs)
                        predicted_risk = risk_label_encoder.inverse_transform([pred])[0] if risk_label_encoder else str(pred)
                except Exception:
                    predicted_risk = "N/A"

            recommended_treatment = "N/A"
            if treatment_model is not None:
                try:
                    prediction_encoded = treatment_model.predict(pd.DataFrame([features]))[0]
                    recommended_treatment = treatment_label_encoder.inverse_transform([prediction_encoded])[0]
                except Exception:
                    recommended_treatment = "N/A"

            summary_data.append({
                "id": p.get("id"),
                "name": p.get("name", "Unknown"),
                "age": p.get("age", "Unknown"),
                "gender": p.get("gender", "Unknown"),
                "condition": cond_desc,
                "predicted_risk": predicted_risk,
                "recommended_treatment": recommended_treatment
            })

        return jsonify(summary_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# Run Flask App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)