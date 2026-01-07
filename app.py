import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("titanic_model.pkl")

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic – przewidywanie przeżycia")
st.write("Podaj dane pasażera, aby przewidzieć czy przeżyje.")

# User input
pclass = st.selectbox("Klasa biletu", [1, 2, 3])
sex = st.selectbox("Płeć", ["male", "female"])
age = st.slider("Wiek", 0, 80, 30)
fare = st.slider("Cena biletu", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port zaokrętowania", ["S", "C", "Q"])

# Encode inputs
sex_encoded = 0 if sex == "male" else 1
embarked_encoded = {"S": 0, "C": 1, "Q": 2}[embarked]

features = np.array([[pclass, sex_encoded, age, fare, embarked_encoded]])

# Prediction
if st.button("🔮 Przewiduj"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"✅ Pasażer **przeżyje** (prawdopodobieństwo: {probability:.2%})")
    else:
        st.error(f"❌ Pasażer **nie przeżyje** (prawdopodobieństwo: {probability:.2%})")
