import streamlit as st
import numpy as np
import joblib

# ======================
# LOAD MODEL
# ======================
rf_model = joblib.load("model/rf_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Jika pakai normalisasi
# scaler = joblib.load("model/scaler.pkl")

st.title("🫀 Heartbeat Classification")
st.write("Prediksi kondisi detak jantung: **Normal / Abnormal**")

# ======================
# INPUT DATA
# ======================
st.subheader("Input Sinyal ECG")

input_text = st.text_area(
    "Masukkan data sinyal ECG (pisahkan dengan koma)",
    "0.01,0.02,0.03,0.01,0.00"
)

if st.button("Prediksi"):
    try:
        # Convert input
        signal = np.array([float(x) for x in input_text.split(",")]).reshape(1, -1)

        # Jika pakai normalisasi
        # signal = scaler.transform(signal)

        # Predict
        prediction = rf_model.predict(signal)
        label = label_encoder.inverse_transform(prediction)[0]

        st.success(f"Hasil Prediksi: **{label.upper()}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan input: {e}")
