import streamlit as st
import pandas as pd
import joblib
from utils.viz import plot_feature_importance
from predict import predict_yield

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")

st.title("🌾 Crop Yield Predictor")
st.write("Predict crop yield based on climate variables using ML models.")

with st.sidebar:
    st.header("Input Parameters")
    crop = st.selectbox("Crop", ["Rice", "Wheat", "Maize"])
    temperature = st.slider("Temperature (°C)", 10.0, 45.0, 25.0)
    rainfall = st.slider("Rainfall (mm)", 0.0, 400.0, 100.0)
    humidity = st.slider("Humidity (%)", 10.0, 100.0, 60.0)
    model_choice = st.selectbox("Model", ["Stacked", "Random Forest", "Linear", "Decision Tree", "Neural Net"])

    if st.button("Predict"):
        input_df = pd.DataFrame({
            "Crop": [crop],
            "Temperature": [temperature],
            "Rainfall": [rainfall],
            "Humidity": [humidity]
        })
        yield_prediction = predict_yield(input_df, model_choice)
        st.success(f"Predicted Yield: {yield_prediction:.2f} tons/hectare")
        st.pyplot(plot_feature_importance())