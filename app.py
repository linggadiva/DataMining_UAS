import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv("produk_klaster.csv")

st.title("Prediksi dan Visualisasi Klaster Produk")

st.markdown("Masukkan fitur produk untuk memprediksi klasternya.")

# Cek apakah kolom klaster ada
if "klaster" not in df.columns:
    st.warning("Kolom 'klaster' tidak ditemukan di CSV, visualisasi terbatas.")
    df["klaster"] = -1  # Tambahkan dummy cluster

# Ambil kolom fitur (tanpa kolom klaster)
fitur_input = df.drop(columns=["klaster"], errors="ignore").columns.tolist()

# Input fitur dari user
user_input = {}
for fitur in fitur_input:
    nilai_min = float(df[fitur].min())
    nilai_max = float(df[fitur].max())
    nilai_mean = float(df[fitur].mean())
    user_input[fitur] = st.number_input(
        label=f"{fitur}",
        min_value=nilai_min,
        max_value=nilai_max,
        value=nilai_mean
    )

# Tombol prediksi
if st.button("Prediksi Klaster"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)
    st.success(f"Produk diprediksi masuk ke dalam klaster: **{pred[0]}**")

# Visualisasi data
st.header("Visualisasi Klaster")

# Pilih fitur untuk visualisasi
fitur_x = st.selectbox("Pilih fitur untuk sumbu X", fitur_input, index=0)
fitur_y = st.selectbox("Pilih fitur untuk sumbu Y", fitur_input, index=1)

fig, ax = plt.subplots()
sns.scatterplot(data=df, x=fitur_x, y=fitur_y, hue="klaster", palette="Set2", ax=ax)
ax.set_title("Visualisasi Klaster Berdasarkan Fitur")
st.pyplot(fig)
