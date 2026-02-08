import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import joblib
import os

# Import Class Preprocessor yang sudah kita buat
from src.preprocessor import TextPreprocessor

# Import Class Preprocessor yang sudah kita buat
from src.preprocessor import TextPreprocessor

# --- 1. LOAD RESOURCES (Hanya sekali saat app start) ---
@st.cache_resource
def load_prediction_resources():
    # Load Robot Pembersih
    # Pastikan path kamus alay benar
    cleaner = TextPreprocessor(slang_path=os.path.join("assets/cleaning", "new_kamusalay.csv"))
    
    # Load Otak (Model) & Penerjemah (Vectorizer)
    # Pastikan file .pkl sudah ada di folder assets
    try:
        model = joblib.load(os.path.join("assets/model", "svm_model.pkl"))
        vectorizer = joblib.load(os.path.join("assets/model", "tfidf_vectorizer.pkl"))
        return cleaner, model, vectorizer
    except FileNotFoundError:
        return None, None, None

cleaner, model, vectorizer = load_prediction_resources()

st.set_page_config(page_title="Analisis Sentimen Rumah Sakit Semarang", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("dataset/data_labeling_sentiment_emotion.csv")

df = load_data()

st.title("🏥 Dashboard Analisis Sentimen Rumah Sakit di Semarang")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analisis Sentimen",
    "😊 Analisis Emosi",
    "🗺️ Wordcloud Sentimen",
    "🔤 Wordcloud Emosi"
])

with tab1:
    st.header("📊 Perbandingan Sentimen Keseluruhan")
    sent_count = df["sentiment"].value_counts()
    fig = px.pie(values=sent_count.values, names=sent_count.index, title="Proporsi Sentimen")
    st.plotly_chart(fig, use_container_width=True)

    st.header("📈 Sentimen per Rumah Sakit")
    fig2 = px.histogram(df, x="rumah_sakit", color="sentiment", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    st.header("🏥 Sentimen Berdasarkan Tipe Rumah Sakit")
    fig3 = px.histogram(df, x="tipe_rs", color="sentiment", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("📌 Ringkasan Sentimen per Tipe Rumah Sakit")
    summary = df.groupby(["tipe_rs", "sentiment"]).size().reset_index(name="jumlah")
    st.dataframe(summary)
    
    st.subheader("📈 Persentase Sentimen per Tipe Rumah Sakit")

    # --- KODE BARU (CARA AMAN) ---
    
    # 1. Hitung jumlah per grup (Tipe RS & Sentimen)
    # reset_index langsung mengubahnya menjadi DataFrame biasa, bukan MultiIndex
    df_percent = df.groupby(["tipe_rs", "sentiment"]).size().reset_index(name="jumlah")
    
    # 2. Hitung total per Tipe RS untuk pembagi (menggunakan transform agar dimensi data tetap sama)
    df_percent["total"] = df_percent.groupby("tipe_rs")["jumlah"].transform("sum")
    
    # 3. Hitung persentase manual
    df_percent["persentase"] = (df_percent["jumlah"] / df_percent["total"]) * 100
    
    # Tampilkan tabel kecil untuk pengecekan
    st.write(df_percent[["tipe_rs", "sentiment", "persentase"]].head())

    # Plotting
    fig_pct = px.bar(
        df_percent,
        x="tipe_rs",
        y="persentase",
        color="sentiment",
        barmode="group",
        title="Persentase Sentimen Berdasarkan Tipe Rumah Sakit (%)"
    )
    st.plotly_chart(fig_pct, use_container_width=True)

    st.subheader("🔍 Analisis Mendalam per Tipe Rumah Sakit")
    tipe_selected = st.selectbox("Pilih Tipe RS:", sorted(df["tipe_rs"].unique()))

    df_tipe = df[df["tipe_rs"] == tipe_selected]

    st.write(f"Jumlah ulasan untuk tipe {tipe_selected}: {len(df_tipe)}")

    fig_deep = px.pie(
        df_tipe,
        names="sentiment",
        title=f"Proporsi Sentimen untuk RS Tipe {tipe_selected}"
    )
    st.plotly_chart(fig_deep, use_container_width=True)


with tab2:
    st.header("😊 Distribusi Emosi Keseluruhan")
    emos_count = df["emotion"].value_counts()
    fig4 = px.bar(x=emos_count.index, y=emos_count.values, title="Distribusi Emosi")
    st.plotly_chart(fig4, use_container_width=True)

    st.header("📌 Emosi per Rumah Sakit")
    fig5 = px.histogram(df, x="rumah_sakit", color="emotion", barmode="group")
    st.plotly_chart(fig5, use_container_width=True)
    
    st.subheader("📌 Emosi Berdasarkan Tipe Rumah Sakit")
    tipe_selected_emos = st.selectbox("Pilih Tipe RS untuk Emosi:", sorted(df["tipe_rs"].unique()))

    df_emos_tipe = df[df["tipe_rs"] == tipe_selected_emos]

    fig_emos_tipe = px.bar(
        df_emos_tipe["emotion"].value_counts(),
        title=f"Distribusi Emosi untuk RS Tipe {tipe_selected_emos}"
    )
    st.plotly_chart(fig_emos_tipe, use_container_width=True)


with tab3:
    st.header("🔤 Wordcloud Berdasarkan Sentimen")
    selected_sent = st.selectbox("Pilih Sentimen", df["sentiment"].unique())

    wc_text = " ".join(df[df["sentiment"] == selected_sent]["ulasan_stopwords"].astype(str))

    wc = WordCloud(background_color="white").generate(wc_text)
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(plt)
    
with tab4:
    st.header("🔤 Wordcloud Berdasarkan Emosi")
    selected_sent = st.selectbox("Pilih Emosi", df["emotion"].unique())

    wc_text = " ".join(df[df["emotion"] == selected_sent]["ulasan_stopwords"].astype(str))

    wc = WordCloud(background_color="white").generate(wc_text)
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(plt)

st.sidebar.header("📄 Export Laporan")

if st.sidebar.button("Download PDF"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, 750, "Laporan Analisis Sentimen Rumah Sakit Semarang")

    c.setFont("Helvetica", 12)
    c.drawString(30, 720, f"Total Data: {len(df)} ulasan")

    c.drawString(30, 690, "Top Sentimen:")
    y = 670
    for label, val in df["sentiment"].value_counts().items():
        c.drawString(40, y, f"{label}: {val}")
        y -= 20

    c.save()
    buffer.seek(0)
    st.sidebar.download_button("Download PDF", buffer, "laporan_sentimen.pdf")

# --- 2. AREA PREDIKSI (UI) ---
st.sidebar.markdown("---")
st.sidebar.header("🤖 Prediksi Sentimen Baru")

if model is not None and vectorizer is not None:
    # Input User
    user_input = st.sidebar.text_area("Masukkan ulasan pengalaman RS:", height=100)
    
    if st.sidebar.button("Analisis Ulasan"):
        if user_input.strip() == "":
            st.sidebar.warning("Mohon isi ulasan terlebih dahulu.")
        else:
            # A. PREPROCESSING (Diam-diam di belakang layar)
            clean_text = cleaner.clean_text(user_input, use_stemming=True)
            
            # B. VECTORIZATION (Ubah teks jadi angka)
            # transform butuh input berupa list, makanya dikurung []
            text_vector = vectorizer.transform([clean_text])
            
            # C. PREDICTION (Model menebak)
            prediction = model.predict(text_vector)[0]
            
            # D. TAMPILKAN HASIL
            st.sidebar.markdown("### Hasil Analisis:")
            
            if prediction == 2: # Sesuaikan dengan label output model Anda (bisa 0/1 atau 'negatif'/'positif')
                st.sidebar.success(f"**Sentimen: POSITIF** 😊")
            elif prediction == 0:
                st.sidebar.error(f"**Sentimen: NEGATIF** 😡")
            else:
                st.sidebar.info(f"**Sentimen: NETRAL** 😐")
                
            # (Opsional) Tampilkan teks bersih untuk debug (bisa dihapus nanti)
            with st.expander("Lihat Teks yang Diproses"):
                st.write(clean_text)
else:
    st.sidebar.error("File Model/Vectorizer belum ditemukan di folder 'assets/'. Harap simpan model dari Notebook terlebih dahulu.")
