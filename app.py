import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hospital Sentiment Dashboard", layout="wide")

# --------------------
# Load Data
# --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/dataset_rumah_sakit_final.csv")
    return df

df = load_data()

st.title("📊 Dashboard Sentimen & Emosi Rumah Sakit di Semarang")

# ============================================================
# 1. SUMMARY
# ============================================================
st.header("📌 Ringkasan Keseluruhan Kota Semarang")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Ulasan", len(df))

with col2:
    st.metric("Jumlah Rumah Sakit", df["rumah_sakit"].nunique())

with col3:
    st.metric("Jenis Emosi Unik", df["emotion_label"].nunique())

# -------------------- Sentiment Distribution --------------------
st.subheader("Distribusi Sentimen (Keseluruhan)")
sentiment_counts = df["sentiment_label_final"].value_counts()

fig_sent = px.bar(
    sentiment_counts,
    title="Distribusi Sentimen",
    labels={"index": "Sentimen", "value": "Jumlah"},
    color=sentiment_counts.index
)
st.plotly_chart(fig_sent, use_container_width=True)

# -------------------- Emotion Distribution --------------------
st.subheader("Distribusi Emosi (Keseluruhan)")
emotion_counts = df["emotion_label"].value_counts()

fig_em = px.bar(
    emotion_counts,
    title="Distribusi Emosi",
    labels={"index": "Emosi", "value": "Jumlah"},
    color=emotion_counts.index
)
st.plotly_chart(fig_em, use_container_width=True)

# ============================================================
# 2. ANALISIS PER RUMAH SAKIT
# ============================================================
st.header("🏥 Analisis per Rumah Sakit")

rs_list = sorted(df["rumah_sakit"].unique())
selected_rs = st.selectbox("Pilih Rumah Sakit:", rs_list)

df_rs = df[df["rumah_sakit"] == selected_rs]

colA, colB = st.columns(2)

with colA:
    st.subheader("Sentimen pada RS terpilih")
    fig_rs_sent = px.bar(
        df_rs["sentiment_label_final"].value_counts(),
        title=f"Distribusi Sentimen - {selected_rs}",
        labels={"index": "Sentimen", "value": "Jumlah"},
        color=df_rs["sentiment_label_final"].unique()
    )
    st.plotly_chart(fig_rs_sent, use_container_width=True)

with colB:
    st.subheader("Emosi pada RS terpilih")
    fig_rs_emo = px.bar(
        df_rs["emotion_label"].value_counts(),
        title=f"Distribusi Emosi - {selected_rs}",
        labels={"index": "Emosi", "value": "Jumlah"},
        color=df_rs["emotion_label"].unique()
    )
    st.plotly_chart(fig_rs_emo, use_container_width=True)

# ============================================================
# 3. WORDCLOUD SENTIMEN
# ============================================================
st.header("☁ WordCloud berdasarkan Sentimen / Emosi")

option = st.selectbox("Pilih berdasarkan:", ["Sentimen", "Emosi"])

if option == "Sentimen":
    chosen = st.selectbox("Pilih sentimen:", df["sentiment_label_final"].unique())
    text = " ".join(df[df["sentiment_label_final"] == chosen]["ulasan_casefolding"])
else:
    chosen = st.selectbox("Pilih emosi:", df["emotion_label"].unique())
    text = " ".join(df[df["emotion_label"] == chosen]["ulasan_casefolding"])

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

fig_wc = plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

st.pyplot(fig_wc)

# ============================================================
# 4. TABEL DETAIL
# ============================================================
st.header("📄 Data Tabel Ulasan")
st.dataframe(df)