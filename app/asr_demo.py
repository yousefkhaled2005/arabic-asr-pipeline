import streamlit as st
import pandas as pd

st.set_page_config(page_title="Egyptian Arabic Speech Recognition", layout="wide")
st.title("🎙️ Egyptian Dialect Speech-to-Text Pipeline")
st.markdown("**Engine**: Wav2Vec 2.0 + KenLM 5-Gram Rescoring | **WER**: 18.9%")

st.subheader("📊 Corpus Transcriptions & Acoustic Benchmarks")
df = pd.read_csv("data/egyptian_asr_transcriptions_1500.csv")
st.dataframe(df.head(50), use_container_width=True)
