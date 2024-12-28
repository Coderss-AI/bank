import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

model = joblib.load("bank_model.pkl")

st.header("Kredit Tasdiqlash")

st.markdown("Ma'lumotlaringizni kiriting:")
age = st.slider("Yosh:", 18, 100, 25)
income = st.slider("Yillik daromad (so'm):", 10000, 1000000, 50000)
loan_amount = st.slider("Kredit miqdori (so'm):", 1000, 500000, 20000)
credit_score = st.slider("Kredit reytingi:", 300, 850, 700)

if st.button("🔮 Aniqlash"):
    features = np.array([[age, income, loan_amount, credit_score]])
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("✅ Kredit tasdiqlandi!")
    else:
        st.error("❌ Kredit rad etildi.")
