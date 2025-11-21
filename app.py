import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("🌾 AgroAI Recommender – Intelligent Crop Recommendation System")

@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

@st.cache_resource
def train_model(df):
    X = df.drop('label', axis=1)
    y = df['label']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

df = load_data()
st.subheader("Dataset Overview")
st.write(df.head())

model = train_model(df)

st.subheader("Enter Soil & Weather Conditions")

N = st.number_input("Nitrogen (N)", 0, 200, 50)
P = st.number_input("Phosphorus (P)", 0, 200, 50)
K = st.number_input("Potassium (K)", 0, 200, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 100.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
pH = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)

if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"🌱 Recommended Crop: **{prediction[0]}**")
