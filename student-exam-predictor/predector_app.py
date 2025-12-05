
# Streamlit predictor app for "Predicting Student Exam Success"

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# -----------------------
# Constants / config
# -----------------------
CSV_PATH = "courses/student_exam.csv"   # path to the dataset produced in Lesson1
MODEL_FILE = "courses/logreg.joblib"
SCALER_FILE = "courses/scaler.joblib"
FEATURES_FILE = "courses/feature_names.joblib"

# -----------------------
# Helper: train model if not present
# -----------------------
def train_model_from_csv(csv_path=CSV_PATH):
    """Train a logistic regression and a StandardScaler on the CSV and return (model, scaler, feature_names)."""
    df = pd.read_csv(csv_path)

    # Select features available before final grade (avoid G3)
    X = df[['sex','studytime','failures','absences','G1','G2']].copy()
    y = df['passed']

    # Encode categorical 'sex'
    X = pd.get_dummies(X, columns=['sex'], drop_first=True)  # creates column 'sex_M' if applicable

    # Numeric columns to scale
    numeric_cols = ['studytime','failures','absences','G1','G2']

    # Fit scaler on numeric columns
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Fit logistic regression on scaled features
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_scaled, y)

    feature_names = X_scaled.columns.tolist()
    return model, scaler, feature_names

@st.cache_resource
def get_or_train_model():
    """Load model/scaler if present, otherwise train and save them."""
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(FEATURES_FILE):
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            feature_names = joblib.load(FEATURES_FILE)
        else:
            model, scaler, feature_names = train_model_from_csv()
            joblib.dump(model, MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
            joblib.dump(feature_names, FEATURES_FILE)
    except Exception as e:
        # If anything fails, train from CSV as fallback
        model, scaler, feature_names = train_model_from_csv()
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(feature_names, FEATURES_FILE)

    return model, scaler, feature_names

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Student Exam Predictor", layout="centered")
st.title("Predicting Student Exam Success — Mini App")
st.markdown(
    """
This simple app predicts whether a student will pass the exam (`passed = 1`) based on features
available before the final grade: sex, study time, past failures, absences, and intermediate grades G1 & G2.

The model is a Logistic Regression trained on `student_exam.csv` (or loaded if a saved model exists).
"""
)

model, scaler, feature_names = get_or_train_model()

# Sidebar inputs
st.sidebar.header("Student inputs")
sex_input = st.sidebar.radio("Sex", ("M", "F"), index=1)
studytime_input = st.sidebar.selectbox("Study time (1:<2h, 2:2-5h, 3:5-10h, 4:>10h)", [1,2,3,4], index=1)
failures_input = st.sidebar.slider("Past failures (0-3)", min_value=0, max_value=3, value=0)
absences_input = st.sidebar.slider("Absences (0-30)", min_value=0, max_value=30, value=3)
G1_input = st.sidebar.slider("G1 (0-20)", min_value=0, max_value=20, value=11)
G2_input = st.sidebar.slider("G2 (0-20)", min_value=0, max_value=20, value=11)

# Optional threshold control
threshold = st.sidebar.slider("Decision threshold (probability to predict PASS)", 0.0, 1.0, 0.5, 0.05)

# Build a single-row DataFrame from inputs
input_df = pd.DataFrame({
    'studytime': [studytime_input],
    'failures': [failures_input],
    'absences': [absences_input],
    'G1': [G1_input],
    'G2': [G2_input],
    'sex': [sex_input]
})

# Encode sex as in training
input_df = pd.get_dummies(input_df, columns=['sex'], drop_first=True)
# Ensure feature columns match training features (create missing dummy columns with 0)
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
# Reorder columns to match training order
input_df = input_df[feature_names]

# Scale numeric columns using the loaded scaler
numeric_cols = ['studytime','failures','absences','G1','G2']  # same as training
input_scaled = input_df.copy()
input_scaled[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Predict probability and class
proba_pass = model.predict_proba(input_scaled)[0, 1]   # probability for class '1' (passed)
pred_class = int(proba_pass >= threshold)             # apply threshold to decide pass/fail

# Display results
st.subheader("Prediction")
st.write(f"**Predicted class:** {'PASS' if pred_class==1 else 'FAIL'}")
st.write(f"**Probability of PASS:** {proba_pass:.3f} (threshold = {threshold:.2f})")

# Simple explanation: per-feature contribution to the logit
coefs = model.coef_[0]         # coefficients correspond to feature_names order
scaled_values = input_scaled.iloc[0].values
contributions = coefs * scaled_values
expl_df = pd.DataFrame({
    'feature': feature_names,
    'coef': coefs,
    'scaled_value': scaled_values,
    'contribution': contributions
})
expl_df['abs_contrib'] = expl_df['contribution'].abs()
expl_df = expl_df.sort_values('abs_contrib', ascending=False).reset_index(drop=True)

st.subheader("Feature contributions (approx.)")
st.write("The contributions are `coef * scaled_value`. Positive values push the model towards PASS, negative towards FAIL.")
st.table(expl_df[['feature','contribution']].round(4))

# Optional bar chart visualization of contributions
st.bar_chart(expl_df.set_index('feature')['contribution'])

# Show the model accuracy on the training data (quick info)
try:
    # For transparency: show training accuracy (re-train on full CSV inside get_or_train_model)
    df_all = pd.read_csv(CSV_PATH)
    X_all = df_all[['sex','studytime','failures','absences','G1','G2']]
    X_all = pd.get_dummies(X_all, columns=['sex'], drop_first=True)
    X_all[numeric_cols] = scaler.transform(X_all[numeric_cols])
    y_all = df_all['passed']
    acc_all = model.score(X_all, y_all)
    st.info(f"Model accuracy on the full dataset (for demo): {acc_all:.3f}")
except Exception:
    pass

st.markdown("""
**Notes**
- This app is educational: model trained on a tiny, synthetic dataset.
- In practice, pretrain and validate models carefully and check fairness before deploying.
""")
