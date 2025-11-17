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

st.set_page_config(page_title="Analisis Sentimen Rumah Sakit Semarang", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("dataset/dataset_rumah_sakit_final.csv")

df = load_data()

st.title("🏥 Dashboard Analisis Sentimen Rumah Sakit di Semarang")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analisis Sentimen",
    "😊 Analisis Emosi",
    "🗺️ Peta Rumah Sakit",
    "🔤 Wordcloud"
])

with tab1:
    st.header("📊 Perbandingan Sentimen Keseluruhan")
    sent_count = df["sentiment_label_final"].value_counts()
    fig = px.pie(values=sent_count.values, names=sent_count.index, title="Proporsi Sentimen")
    st.plotly_chart(fig, use_container_width=True)

    st.header("📈 Sentimen per Rumah Sakit")
    fig2 = px.histogram(df, x="rumah_sakit", color="sentiment_label_final", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    st.header("🏥 Sentimen Berdasarkan Tipe Rumah Sakit")
    fig3 = px.histogram(df, x="tipe_rs", color="sentiment_label_final", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("📌 Ringkasan Sentimen per Tipe Rumah Sakit")
    summary = df.groupby(["tipe_rs", "sentiment_label_final"]).size().reset_index(name="jumlah")
    st.dataframe(summary)
    
    st.subheader("📈 Persentase Sentimen per Tipe Rumah Sakit")

    df_percent = (
        df.groupby(["tipe_rs", "sentiment_label_final"])
          .size()
          .groupby(level=0)
          .apply(lambda x: 100 * x / x.sum())
          .reset_index(name="persentase")
    )

    fig_pct = px.bar(
        df_percent,
        x="tipe_rs",
        y="persentase",
        color="sentiment_label_final",
        barmode="stack",
        title="Persentase Sentimen Berdasarkan Tipe Rumah Sakit (%)"
    )
    st.plotly_chart(fig_pct, use_container_width=True)

    st.subheader("🔍 Analisis Mendalam per Tipe Rumah Sakit")
    tipe_selected = st.selectbox("Pilih Tipe RS:", sorted(df["tipe_rs"].unique()))

    df_tipe = df[df["tipe_rs"] == tipe_selected]

    st.write(f"Jumlah ulasan untuk tipe {tipe_selected}: {len(df_tipe)}")

    fig_deep = px.pie(
        df_tipe,
        names="sentiment_label_final",
        title=f"Proporsi Sentimen untuk RS Tipe {tipe_selected}"
    )
    st.plotly_chart(fig_deep, use_container_width=True)


with tab2:
    st.header("😊 Distribusi Emosi Keseluruhan")
    emos_count = df["emotion_label"].value_counts()
    fig4 = px.bar(x=emos_count.index, y=emos_count.values, title="Distribusi Emosi")
    st.plotly_chart(fig4, use_container_width=True)

    st.header("📌 Emosi per Rumah Sakit")
    fig5 = px.histogram(df, x="rumah_sakit", color="emotion_label", barmode="group")
    st.plotly_chart(fig5, use_container_width=True)
    
    st.subheader("📌 Emosi Berdasarkan Tipe Rumah Sakit")
    tipe_selected_emos = st.selectbox("Pilih Tipe RS untuk Emosi:", sorted(df["tipe_rs"].unique()))

    df_emos_tipe = df[df["tipe_rs"] == tipe_selected_emos]

    fig_emos_tipe = px.bar(
        df_emos_tipe["emotion_label"].value_counts(),
        title=f"Distribusi Emosi untuk RS Tipe {tipe_selected_emos}"
    )
    st.plotly_chart(fig_emos_tipe, use_container_width=True)

with tab3:
    st.header("🗺️ Peta Rumah Sakit Berdasarkan Sentimen")
    st.info("Peta dapat ditambahkan jika dataset berisi latitude & longitude.")

with tab4:
    st.header("🔤 Wordcloud Berdasarkan Sentimen")
    selected_sent = st.selectbox("Pilih Sentimen", df["sentiment_label_final"].unique())

    wc_text = " ".join(df[df["sentiment_label_final"] == selected_sent]["ulasan_stopwords"].astype(str))

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
    for label, val in df["sentiment_label_final"].value_counts().items():
        c.drawString(40, y, f"{label}: {val}")
        y -= 20

    c.save()
    buffer.seek(0)
    st.sidebar.download_button("Download PDF", buffer, "laporan_sentimen.pdf")
