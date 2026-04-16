import streamlit as st
import pandas as pd

st.title("Sales Data Analyzer")

uploaded_file = st.file_uploader("Upload CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())