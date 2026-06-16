import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Risiko Stroke",
    page_icon="🏥",
    layout="centered"
)

# --- LOAD MODEL & INFO ---
@st.cache_resource
def load_model():
    if os.path.exists('model_terbaik.pkl'):
        with open('model_terbaik.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    return None

@st.cache_data
def load_info():
    if os.path.exists('info_model.json'):
        with open('info_model.json', 'r') as f:
            info = json.load(f)
        return info
    return None

model = load_model()
info_model = load_info()

# --- HEADER APLIKASI ---
st.title("🏥 Aplikasi Prediksi Risiko Stroke")
st.write("Aplikasi ini menggunakan Machine Learning untuk memprediksi risiko stroke berdasarkan faktor kesehatan dan gaya hidup.")
st.markdown("---")

if model is None:
    st.error("⚠️ File `model_terbaik.pkl` tidak ditemukan! Pastikan kamu sudah mengunggah file model hasil training Colab ke repositori GitHub yang sama.")
else:
    # Tampilkan info model jika ada
    if info_model:
        st.sidebar.header("📊 Informasi Model")
        st.sidebar.info(f"**Model Terbaik:** {info_model['nama_model']}\n\n"
                        f"- **Accuracy:** {info_model['accuracy']:.2f}\n"
                        f"- **Precision:** {info_model['precision']:.2f}\n"
                        f"- **Recall:** {info_model['recall']:.2f}\n"
                        f"- **F1-Score:** {info_model['f1_score']:.2f}")

    st.subheader("📝 Masukkan Data Kesehatan")

    # --- FORM INPUT USER ---
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
        age = st.number_input("Usia (Tahun)", min_value=0, max_value=120, value=30)
        hypertension = st.selectbox("Memiliki Hipertensi?", ["Tidak", "Ya"])
        heart_disease = st.selectbox("Memiliki Penyakit Jantung?", ["Tidak", "Ya"])
        ever_married = st.selectbox("Pernah Menikah?", ["Yes", "No"])

    with col2:
        work_type = st.selectbox("Tipe Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        Residence_type = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Kadar Glukosa Rata-rata (mg/dL)", min_value=0.0, value=100.0)
        bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, value=25.0)
        smoking_status = st.selectbox("Status Merokok", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    # --- PREPROCESSING INPUT USER (Sesuai dengan Step 1 EDA) ---
    # 1. Konversi Hipertensi & Penyakit Jantung (0 atau 1)
    hypertension_val = 1 if hypertension == "Ya" else 0
    heart_disease_val = 1 if heart_disease == "Ya" else 0

    # 2. Logika Pembuatan Fitur Baru (Feature Engineering)
    # Fitur age_group
    if age < 18:
        age_group_val = "Anak"
    elif age < 40:
        age_group_val = "Dewasa Muda"
    elif age < 60:
        age_group_val = "Dewasa"
    else:
        age_group_val = "Lansia"

    # Fitur bmi_category
    if bmi < 18.5:
        bmi_category_val = "Underweight"
    elif bmi < 25:
        bmi_category_val = "Normal"
    elif bmi < 30:
        bmi_category_val = "Overweight"
    else:
        bmi_category_val = "Obesitas"

    # 3. Label Encoding Manual (Harus konsisten dengan urutan abjad/fit_transform LabelEncoder di Colab)
    # Pemetaan berdasarkan urutan string unik dari data asli dataset stroke
    encoding_maps = {
        'gender': {'Female': 0, 'Male': 1, 'Other': 2},
        'ever_married': {'No': 0, 'Yes': 1},
        'work_type': {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self-employed': 3, 'children': 4},
        'Residence_type': {'Rural': 0, 'Urban': 1},
        'smoking_status': {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3},
        'age_group': {'Anak': 0, 'Dewasa': 1, 'Dewasa Muda': 2, 'Lansia': 3},
        'bmi_category': {'Normal': 0, 'Obesitas': 1, 'Overweight': 2, 'Underweight': 3}
    }

    # 4. Standardisasi Kolom Numerik (Menggunakan nilai rata-rata & deviasi standar kasar dataset stroke asli)
    # Karena tidak mengekspor object scaler dari Colab, kita gunakan pendekatan manual invers dari Standard Scaler dataset stroke asli
    age_scaled = (age - 43.2266) / 22.6126
    glucose_scaled = (avg_glucose_level - 106.1476) / 45.2835
    bmi_scaled = (bmi - 28.8932) / 7.8540

    # --- TOMBOL PREDIKSI ---
    st.markdown("---")
    if st.button("🔮 Hitung Risiko Stroke", type="primary"):
        
        # Susun dataframe input sesuai dengan nama dan urutan kolom saat training (X)
        input_data = pd.DataFrame([{
            'gender': encoding_maps['gender'][gender],
            'age': age_scaled,
            'hypertension': hypertension_val,
            'heart_disease': heart_disease_val,
            'ever_married': encoding_maps['ever_married'][ever_married],
            'work_type': encoding_maps['work_type'][work_type],
            'Residence_type': encoding_maps['Residence_type'][Residence_type],
            'avg_glucose_level': glucose_scaled,
            'bmi': bmi_scaled,
            'smoking_status': encoding_maps['smoking_status'][smoking_status],
            'age_group': encoding_maps['age_group'][age_group_val],
            'bmi_category': encoding_maps['bmi_category'][bmi_category_val]
        }])

        # Lakukan prediksi
        prediksi = model.predict(input_data)[0]
        
        # Cek jika model mendukung predict_proba (seperti Random Forest / Decision Tree)
        try:
            probabilitas = model.predict_proba(input_data)[0][1] * 100
        except:
            probabilitas = None

        # --- TAMPILKAN HASIL ---
        st.subheader("📊 Hasil Analisis")
        if prediksi == 1:
            st.error(f"🚨 **RISIKO TINGGI:** Model memprediksi Anda memiliki kecenderungan risiko STROKE.")
            if probabilitas is not None:
                st.write(f"Tingkat keyakinan model: **{probabilitas:.2f}%**")
            st.write("💡 *Saran: Segera konsultasikan kondisi kesehatan Anda ke dokter spesialis untuk pemeriksaan lebih lanjut.*")
        else:
            st.success(f"✅ **RISIKO RENDAH:** Model memprediksi Anda saat ini tidak memiliki indikasi kuat ke arah STROKE.")
            if probabilitas is not None:
                st.write(f"Tingkat keyakinan model sehat: **{100 - probabilitas:.2f}%**")
            st.write("💡 *Tetap pertahankan gaya hidup sehat, konsumsi makanan bergizi, dan rutin berolahraga!*")
