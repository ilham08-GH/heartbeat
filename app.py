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
    # Pastikan path model benar
    model_path = "model/rf_model.pkl"
    encoder_path = "model/label_encoder.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        return None, None
    
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

# Load model di awal
rf_model, label_encoder = load_models()

# --- TAMPILAN UTAMA ---
st.title("🫀 Heartbeat Classification")
st.write("""
Aplikasi ini menggunakan **Random Forest** untuk mendeteksi anomali pada suara detak jantung.
Model dilatih menggunakan data *Spectrogram/Audio* yang di-flatten (resolusi tinggi).
""")

# Cek apakah model berhasil di-load
if rf_model is None:
    st.error("⚠️ Model tidak ditemukan! Pastikan file 'rf_model.pkl' dan 'label_encoder.pkl' ada di folder 'model/'.")
    st.stop()

# --- INPUT DATA ---
st.subheader("1. Upload Data Sinyal")
st.info("Karena model membutuhkan input resolusi tinggi (24.705 fitur), silakan upload file `.csv` atau `.txt` berisi data sinyal.")

uploaded_file = st.file_uploader("Pilih file data", type=["csv", "txt"])

# --- PREDIKSI ---
if uploaded_file is not None:
    try:
        # Baca file
        # Asumsi: File berisi angka-angka yang dipisahkan koma, tanpa header
        df_input = pd.read_csv(uploaded_file, header=None)
        
        # Flattening data jika bentuknya masih matriks/banyak baris
        # Kita ubah menjadi 1 baris panjang (1D array)
        data_values = df_input.values.flatten().reshape(1, -1)
        
        st.write("---")
        st.subheader("2. Analisis Data")
        st.write(f"**Ukuran data input:** {data_values.shape[1]} fitur")
        
        # Validasi Input Model
        expected_features = rf_model.n_features_in_
        
        if data_values.shape[1] != expected_features:
            st.error(f"❌ Error: Model mengharapkan **{expected_features}** fitur, tapi file Anda memiliki **{data_values.shape[1]}** fitur.")
            st.warning("Tips: Pastikan file yang diupload adalah data mentah yang sama formatnya dengan data training (sebelum dipotong/dikurangi).")
        else:
            # Tombol Prediksi
            if st.button("Jalankan Prediksi", type="primary"):
                with st.spinner('Sedang menganalisis pola detak jantung...'):
                    # Prediksi
                    prediction_index = rf_model.predict(data_values)[0]
                    prediction_label = label_encoder.inverse_transform([prediction_index])[0]
                    
                    # Probabilitas (Keyakinan Model)
                    prediction_proba = rf_model.predict_proba(data_values)
                    confidence = np.max(prediction_proba) * 100
                
                # Tampilkan Hasil
                st.write("---")
                st.subheader("3. Hasil Diagnosa")
                
                if prediction_label.lower() == 'normal':
                    st.success(f"### ✅ Kondisi: {prediction_label.upper()}")
                else:
                    st.error(f"### ⚠️ Kondisi: {prediction_label.upper()}")
                
                st.metric(label="Tingkat Keyakinan Model", value=f"{confidence:.2f}%")
                
                # Visualisasi Sederhana (Opsional - ambil 100 titik pertama)
                st.line_chart(data_values[0][:100])
                st.caption("Grafik 100 titik data pertama sinyal")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")

# --- SIDEBAR (Info) ---
with st.sidebar:
    st.header("Tentang Model")
    st.text(f"Tipe: Random Forest Classifier")
    if rf_model:
        st.text(f"Fitur Input: {rf_model.n_features_in_}")
    st.markdown("---")
    st.write("Dibuat untuk tugas klasifikasi Time Series.")
