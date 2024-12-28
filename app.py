import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

model = joblib.load("bank_model.pkl")

st.header("Kredit Tasdiqlash")

st.markdown("Ma'lumotlaringizni kiriting:")
age = st.number_input("Yosh:", 18, 100, 25)
income = st.number_input("Yillik daromad ($):", 10000, 1000000, 50000)
loan_amount = st.number_input("Kredit miqdori ($):", 1000, 500000, 20000)
credit_score = st.number_input("Kredit reytingi:", 300, 850, 700)

if st.button("🔮 Aniqlash"):
    features = np.array([[age, income, loan_amount, credit_score]])
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("✅ Kredit tasdiqlandi!")
    else:
        st.error("❌ Kredit rad etildi.")
