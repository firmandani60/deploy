import streamlit as st
import pandas as pd
import joblib

# === Load model dan scaler ===
model = joblib.load("model_ilpd_knn.pkl")
scaler = joblib.load("scaler_ilpd.pkl")

# === Judul aplikasi ===
st.title("🩺 Prediksi Pasien Liver Menggunakan Model ML")

st.markdown("Silakan masukkan data pasien di bawah ini:")

# === Form Input (Pastikan nama kolom sesuai dengan training) ===
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=45)
total_bilirubin = st.number_input("TB (Total Bilirubin)", value=1.0)
direct_bilirubin = st.number_input("DB (Direct Bilirubin)", value=0.5)
alkphos = st.number_input("ALKPHOS (Alkaline Phosphotase)", value=200)
sgpt = st.number_input("SGPT (Alamine Aminotransferase)", value=40)
sgot = st.number_input("SGOT (Aspartate Aminotransferase)", value=50)
tp = st.number_input("TP (Total Protein)", value=6.5)
alb = st.number_input("ALB (Albumin)", value=3.0)
ag_ratio = st.number_input("A/G Ratio", value=1.0)

# Tombol Prediksi
if st.button("🔍 Prediksi Status Pasien"):
    # Mapping gender
    gender_encoded = 1 if gender == "Male" else 0

    # Buat DataFrame input
    input_df = pd.DataFrame([{
        "Gender": gender_encoded,
        "Age": age,
        "TB": total_bilirubin,
        "DB": direct_bilirubin,
        "ALKPHOS": alkphos,
        "SGPT": sgpt,
        "SGOT": sgot,
        "TP": tp,
        "ALB": alb,
        "A/G Ratio": ag_ratio
    }])

    # Scaling data input
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    label = "Pasien Liver" if prediction == 1 else "Pasien Normal"

    st.success(f"Hasil Prediksi: **{label}**")
