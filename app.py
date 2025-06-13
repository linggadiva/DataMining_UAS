
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dashboard Analisis Produk", layout="wide")

st.title("🛍️ Dashboard Analisis Produk E-Commerce")

uploaded_file = st.file_uploader("📁 Unggah file CSV data produk", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Pembersihan dan transformasi data
    df = df[['product_name', 'product_category_tree', 'retail_price', 'discounted_price', 'product_rating']]
    df.columns = ['nama_produk','jenis_produk', 'harga_retail', 'harga_diskon', 'rating']
    df['harga_retail'] = pd.to_numeric(df['harga_retail'], errors='coerce')
    df['harga_diskon'] = pd.to_numeric(df['harga_diskon'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna()
    df = df[df['harga_diskon'] > 0]
    df['diskon'] = df['harga_retail'] - df['harga_diskon']
    df['best_seller'] = ((df['harga_diskon'] < df['harga_diskon'].median()) & (df['rating'] >= 4)).astype(int)

    # Clustering
    X_cluster = df[['harga_diskon', 'rating', 'diskon']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['klaster'] = kmeans.fit_predict(X_scaled)

    st.sidebar.header("📌 Navigasi")
    page = st.sidebar.radio("Pilih Halaman:", ["Dashboard Klasterisasi", "Prediksi Best Seller", "Tabel Rekomendasi"])

    if page == "Dashboard Klasterisasi":
        st.subheader("📊 Visualisasi Klaster Produk")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df['harga_diskon'], df['rating'], c=df['klaster'], cmap='Set1')
        ax.set_xlabel('Harga Diskon')
        ax.set_ylabel('Rating')
        ax.set_title("Clustering Produk")
        st.pyplot(fig)

        with st.expander("🧾 Deskripsi Klaster"):
            for k in sorted(df['klaster'].unique()):
                desc = df[df['klaster'] == k].describe()
                st.markdown(f"**Klaster {k}**")
                st.write(desc[['harga_diskon', 'rating', 'diskon']])

    elif page == "Prediksi Best Seller":
        st.subheader("📈 Prediksi Produk Baru: Best Seller atau Tidak")
        with st.form("form_prediksi"):
            harga_diskon = st.number_input("Harga Diskon", min_value=0)
            rating = st.slider("Rating Produk", 0.0, 5.0, step=0.1)
            harga_retail = st.number_input("Harga Retail", min_value=0)
            submitted = st.form_submit_button("Prediksi")

        if submitted:
            diskon = harga_retail - harga_diskon
            model = RandomForestClassifier(random_state=42)
            model.fit(df[['harga_diskon', 'rating', 'diskon']], df['best_seller'])
            pred = model.predict([[harga_diskon, rating, diskon]])
            st.success("✅ Produk ini **berpotensi Best Seller!**" if pred[0] == 1 else "⚠️ Produk ini **kurang potensial sebagai Best Seller.**")

    elif page == "Tabel Rekomendasi":
        st.subheader("📋 Produk yang Direkomendasikan untuk Dipromosikan")
        rekomendasi = df[(df['best_seller'] == 0) & (df['rating'] >= 3.5)]
        st.write("🔍 Produk dengan rating cukup tinggi namun belum termasuk best-seller:")
        st.dataframe(rekomendasi[['nama_produk', 'harga_diskon', 'rating', 'klaster']])

        with st.expander("💡 Insight"):
            st.markdown("- Produk dalam **klaster dengan diskon besar dan rating tinggi** cenderung jadi best-seller.")
            st.markdown("- Coba **naikkan rating** dengan meningkatkan kualitas atau ulasan.")
