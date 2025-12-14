import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="Heartbeat Classification",
    page_icon="🫀",
    layout="wide"  # Layout wide agar tabel muat
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
st.title("🫀 Heartbeat Classification Dashboard")
st.write("""
Upload file CSV berisi data sinyal detak jantung. 
Aplikasi ini dapat memproses **satu atau banyak pasien** sekaligus.
""")

if rf_model is None:
    st.error("⚠️ File model tidak ditemukan! Pastikan folder `model/` berisi `rf_model.pkl` dan `label_encoder.pkl`.")
    st.stop()

# --- INPUT DATA ---
st.sidebar.header("Panel Kontrol")
uploaded_file = st.sidebar.file_uploader("Upload File Data (.csv/.txt)", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        # 1. Baca File
        # Membaca tanpa header karena data mentah biasanya hanya angka
        df_input = pd.read_csv(uploaded_file, header=None)
        
        st.subheader("1. Analisis Struktur Data")
        st.write(f"**Total Data Terbaca:** {df_input.shape[0]} Baris, {df_input.shape[1]} Kolom")

        # 2. Validasi & Reshape Data
        # Model membutuhkan tepat 24.705 fitur per pasien.
        expected_features = rf_model.n_features_in_
        total_points = df_input.size
        
        # Logika Cerdas: Cek apakah total poin data adalah kelipatan dari 24.705
        if total_points % expected_features == 0:
            num_samples = total_points // expected_features
            
            # Reshape data menjadi (Jumlah_Pasien, 24705)
            # Ini akan memisahkan kembali data yang mungkin tergabung
            data_values = df_input.values.flatten().reshape(num_samples, expected_features)
            
            st.success(f"✅ Data Valid! Terdeteksi **{num_samples} sampel pasien** dengan format yang benar ({expected_features} fitur).")
            
            # 3. Tombol Prediksi
            if st.button("Jalankan Diagnosa", type="primary"):
                with st.spinner(f'Menganalisis {num_samples} data pasien...'):
                    
                    # Lakukan Prediksi untuk SEMUA baris sekaligus
                    predictions_index = rf_model.predict(data_values)
                    probabilities = rf_model.predict_proba(data_values)
                    
                    # Decode Label (Handling Dict vs LabelEncoder)
                    results = []
                    for i, pred_idx in enumerate(predictions_index):
                        # Ambil Label
                        if isinstance(label_encoder, dict):
                            reverse_mapping = {v: k for k, v in label_encoder.items()}
                            label_str = reverse_mapping.get(pred_idx, "Unknown")
                        else:
                            label_str = label_encoder.inverse_transform([pred_idx])[0]
                        
                        # Ambil Confidence Score (Probabilitas tertinggi)
                        conf_score = np.max(probabilities[i]) * 100
                        
                        results.append({
                            "Pasien Ke": i + 1,
                            "Diagnosa": str(label_str).upper(),
                            "Keyakinan (%)": f"{conf_score:.2f}%",
                            "Status": "✅ Normal" if str(label_str).lower() == 'normal' else "⚠️ Abnormal"
                        })
                    
                    # 4. Tampilkan Hasil dalam Tabel
                    st.write("---")
                    st.subheader("2. Hasil Diagnosa")
                    
                    # Buat DataFrame hasil agar rapi
                    df_results = pd.DataFrame(results)
                    
                    # Highlight warna untuk Abnormal
                    def highlight_abnormal(val):
                        color = 'red' if val == '⚠️ Abnormal' else 'green'
                        return f'color: {color}; font-weight: bold'

                    st.dataframe(df_results.style.applymap(highlight_abnormal, subset=['Status']), use_container_width=True)

                    # Visualisasi Sinyal Pasien Pertama (Contoh)
                    st.write("---")
                    st.subheader("3. Visualisasi Sampel Sinyal (Pasien #1)")
                    st.line_chart(data_values[0][:200]) # 200 titik pertama
                    st.caption("Grafik gelombang 200 titik pertama dari Pasien ke-1.")

        else:
            # Jika dimensinya tidak pas kelipatan 24.705
            st.error(f"❌ Error Dimensi Fatal!")
            st.write(f"Total titik data: **{total_points}**")
            st.write(f"Model membutuhkan kelipatan dari: **{expected_features}**")
            st.warning("Tips: Pastikan file CSV Anda berisi baris-baris data dengan panjang 24.705 kolom (atau 405x61).")

    except Exception as e:
        st.error(f"Terjadi kesalahan sistem: {e}")

else:
    st.info("👈 Silakan upload file CSV di panel sebelah kiri untuk memulai.")
