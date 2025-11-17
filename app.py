# ====================================================
# IMPORT LIBRARY & LOAD DATA
# ====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import io
import json


# Caching model agar tidak loading ulang
@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli"
    )

    emotion_model = pipeline(
        "zero-shot-classification",
        model="SamLowe/roberta-base-go-emotions"
    )

    return sentiment_model, emotion_model

sentiment_model, emotion_model = load_models()

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_rumah_sakit_final.csv")
    return df

df = load_data()

# ====================================================
# SIDEBAR MENU
# ====================================================

st.sidebar.title("📊 Dashboard Analisis Rumah Sakit Semarang")

menu = st.sidebar.radio(
    "Navigasi",
    [
        "Beranda",
        "Statistik Sentimen",
        "Statistik Emosi",
        "Peta Rumah Sakit",
        "Wordcloud",
        "Analisis Tipe RS",
        "Prediksi Manual",
        "Generate Laporan PDF"
    ]
)

# ====================================================
# BERANDA
# ====================================================

if menu == "Beranda":
    st.title("🏥 Dashboard Analisis Sentimen & Emosi Ulasan Rumah Sakit di Semarang")
    st.write("Dataset: Google Maps Reviews (Tahun 2024)")
    st.write("Peneliti: Rio Lintang – Informatika")
    st.write("---")
    st.write("Silakan pilih menu di sidebar untuk melihat analisis.")
    
# ====================================================
# STATISTIK SENTIMEN
# 1. Distribusi sentimen keseluruhan
# 2. Distribusi sentimen per rumah sakit
# ====================================================

if menu == "Statistik Sentimen":
    st.title("📊 Statistik Sentimen")

    # ===============================
    # Sentimen keseluruhan
    # ===============================
    st.subheader("1. Distribusi Sentimen Secara Keseluruhan")
    sent_count = df["sentiment_label_final"].value_counts()
    fig = px.pie(
        values=sent_count.values,
        names=sent_count.index,
        title="Distribusi Sentimen"
    )
    st.plotly_chart(fig)

    # ===============================
    # Sentimen per rumah sakit
    # ===============================
    st.subheader("2. Perbandingan Sentimen per Rumah Sakit")
    rs_group = df.groupby(["rumah_sakit", "sentiment_label_final"]).size().reset_index(name="count")

    fig = px.bar(
        rs_group,
        x="rumah_sakit",
        y="count",
        color="sentiment_label_final",
        title="Sentimen per Rumah Sakit",
        barmode="group"
    )
    st.plotly_chart(fig)
    
    
# ====================================================
# STATISTIK EMOSI
# 1. Distribusi emosi per rumah sakit
# ====================================================

if menu == "Statistik Emosi":
    st.title("🎭 Statistik Emosi")

    emo_group = df.groupby(["rumah_sakit", "emotion_label"]).size().reset_index(name="count")

    fig = px.bar(
        emo_group,
        x="rumah_sakit",
        y="count",
        color="emotion_label",
        title="Emosi per Rumah Sakit",
        barmode="stack"
    )
    st.plotly_chart(fig)
    
# ====================================================
# WORDCLOUD SENTIMEN & EMOSI
# 1. Wordcloud per sentimen & emosi
# 2. Analisis sentimen berdasarkan tipe RS (A/B/C/D)
# ====================================================

if menu == "Wordcloud":
    st.title("☁ Wordcloud Kata Dominan")

    pilihan = st.selectbox("Pilih tipe analisis:", ["Sentimen", "Emosi"])

    if pilihan == "Sentimen":
        kategori = st.selectbox("Pilih Sentimen:", df["sentiment_label_final"].unique())
        teks = " ".join(df[df["sentiment_label_final"] == kategori]["ulasan_lemmatize"].astype(str))
    else:
        kategori = st.selectbox("Pilih Emosi:", df["emotion_label"].unique())
        teks = " ".join(df[df["emotion_label"] == kategori]["ulasan_lemmatize"].astype(str))

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(teks)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)
    
# ====================================================
# PREDIKSI SENTIMENT
# ====================================================

# 1. MUAT MODEL
@st.cache_resource
def load_model():
    sentiment_model = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli"
    )

    emotion_model = pipeline(
        "zero-shot-classification",
        model="SamLowe/roberta-base-go-emotions"
    )

    return sentiment_model, emotion_model

sentiment_model, emotion_model = load_model()


# 2. SET LABEL
sentiment_labels = ["positif", "negatif", "netral"]

emotion_labels = [
    "terkejut", "antisipasi", "takut", 
    "marah", "jijik", "yakin", "sedih", "bahagia"
]

# ==============================
# UI INPUT PREDIKSI
# ==============================
st.subheader("🔍 Prediksi Sentimen & Emosi (Manual Input)")

text = st.text_area("Masukkan teks ulasan:", height=150)

if st.button("Prediksi"):
    if text.strip() == "":
        st.warning("Teks masih kosong!")
    else:
        with st.spinner("Memproses..."):

            # Prediksi Sentimen
            sent_result = sentiment_model(
                text,
                candidate_labels=sentiment_labels
            )

            pred_sentiment = sent_result["labels"][0]
            pred_sentiment_score = sent_result["scores"][0]

            # Prediksi Emosi
            emo_result = emotion_model(
                text,
                candidate_labels=emotion_labels
            )

            pred_emotion = emo_result["labels"][0]
            pred_emotion_score = emo_result["scores"][0]

        # ==============================
        # OUTPUT
        # ==============================
        st.success("Hasil Prediksi")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟦 Sentimen")
            st.write(f"**Label:** {pred_sentiment}")
            st.write(f"**Confidence:** {pred_sentiment_score:.4f}")

        with col2:
            st.markdown("### 🟩 Emosi Dominan")
            st.write(f"**Label:** {pred_emotion}")
            st.write(f"**Confidence:** {pred_emotion_score:.4f}")

        # Tabel skor lengkap emosi
        st.markdown("### 📊 Skor Semua Emosi")
        df_emo = pd.DataFrame({
            "Emosi": emo_result["labels"],
            "Skor": emo_result["scores"]
        })
        st.dataframe(df_emo)

        # ==============================
        # DOWNLOAD CSV
        # ==============================
        df_download = pd.DataFrame({
            "text": [text],
            "sentiment": [pred_sentiment],
            "sentiment_score": [pred_sentiment_score],
            "emotion": [pred_emotion],
            "emotion_score": [pred_emotion_score]
        })

        csv = df_download.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Unduh Hasil Prediksi CSV",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )
        

# ==============================
# GENERATE PDF
# ==============================

if menu == "Generate Laporan PDF":
    st.title("📄 Generate Laporan Analisis (PDF)")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Laporan Analisis Sentimen Rumah Sakit Semarang", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Ringkasan Statistik:", styles["Heading2"]))
    story.append(Paragraph(f"Total data: {len(df)} ulasan", styles["BodyText"]))
    story.append(Spacer(1, 12))

    doc.build(story)
    pdf = buffer.getvalue()

    st.download_button(
        "📥 Download PDF",
        pdf,
        file_name="laporan_analisis.pdf",
        mime="application/pdf"
    )