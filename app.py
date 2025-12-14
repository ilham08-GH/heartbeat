import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="Heartbeat Classification",
    page_icon="🫀",
    layout="centered"
)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_models():
    model_path = "model/rf_model.pkl"
    encoder_path = "model/label_encoder.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        return None, None
    
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load resources
rf_model, label_encoder = load_models()

# --- TAMPILAN UTAMA ---
st.title("🫀 Heartbeat Classification")
st.write("""
Aplikasi deteksi anomali detak jantung menggunakan **Random Forest**.
Model ini dilatih dengan data time-series resolusi tinggi (raw audio signal).
""")

if rf_model is None:
    st.error("⚠️ File model tidak ditemukan! Pastikan folder `model/` berisi `rf_model.pkl` dan `label_encoder.pkl`.")
    st.stop()

# --- INPUT DATA ---
st.subheader("1. Upload Data Sinyal")
st.info(f"Model membutuhkan input file CSV/TXT dengan **{rf_model.n_features_in_}** fitur (titik data).")

uploaded_file = st.file_uploader("Upload file data (.csv/.txt)", type=["csv", "txt"])

# --- PROSES & PREDIKSI ---
if uploaded_file is not None:
    try:
        # 1. Baca File
        # Membaca file tanpa header, asumsi data dipisah koma
        df_input = pd.read_csv(uploaded_file, header=None)
        
        # 2. Preprocessing (Flattening)
        # Mengubah data menjadi array 1 baris (1, n_features)
        data_values = df_input.values.flatten().reshape(1, -1)
        
        st.write("---")
        st.subheader("2. Analisis Data")
        st.write(f"**Dimensi data input:** {data_values.shape[1]} titik data")
        
        # 3. Validasi Dimensi
        expected_features = rf_model.n_features_in_
        
        if data_values.shape[1] != expected_features:
            st.error(f"❌ Error Dimensi: Model butuh {expected_features} fitur, tapi file ini punya {data_values.shape[1]}.")
            st.warning("Pastikan Anda mengupload data mentah (raw signal) yang sesuai format training.")
        else:
            # 4. Tombol Prediksi
            if st.button("Jalankan Diagnosa", type="primary"):
                with st.spinner('Menganalisis pola gelombang...'):
                    # Prediksi Kelas
                    prediction_index = rf_model.predict(data_values)[0]
                    
                    # Handling Label Encoder (Support Dict atau Scikit-Learn)
                    if isinstance(label_encoder, dict):
                        # Jika format lama (Dictionary)
                        reverse_mapping = {v: k for k, v in label_encoder.items()}
                        prediction_label = reverse_mapping.get(prediction_index, "Unknown")
                    else:
                        # Jika format baru (LabelEncoder Scikit-Learn)
                        prediction_label = label_encoder.inverse_transform([prediction_index])[0]
                    
                    # Probabilitas
                    prediction_proba = rf_model.predict_proba(data_values)
                    confidence = np.max(prediction_proba) * 100
                
                # 5. Output
                st.write("---")
                st.subheader("3. Hasil Prediksi")
                
                # Tampilan dinamis berdasarkan hasil
                if str(prediction_label).lower() == 'normal':
                    st.success(f"### ✅ Status: {str(prediction_label).upper()}")
                else:
                    st.error(f"### ⚠️ Status: {str(prediction_label).upper()}")
                
                st.metric("Tingkat Keyakinan (Confidence)", f"{confidence:.2f}%")
                
                # Grafik Sinyal (Preview 200 titik pertama)
                st.write("#### Visualisasi Sinyal (Cuplikan Awal)")
                st.line_chart(data_values[0][:200])

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Info Model")
    st.write("Model: Random Forest")
    st.write(f"Input Features: {rf_model.n_features_in_}")
    st.markdown("---")
    st.write("© 2024 Heartbeat Project")
