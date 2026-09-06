import streamlit as st
import pandas as pd 
import os 
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Fixed: File name champion-model.pkl (matching Train.py)
MODEL_PATH = os.path.join(BASE_DIR, "models", "champion-model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please run 'python src/Train.py' first.")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Advertising Sales Predictor")

tv = st.number_input("TV Budget", min_value=0.0, value=100.0)
radio = st.number_input("Radio Budget", min_value=0.0, value=25.0)
newspaper = st.number_input("Newspaper Budget", min_value=0.0, value=10.0)

if st.button("Predict Sales"):
    if model is not None:
        input_data = pd.DataFrame({
            "TV": [tv],
            "Radio": [radio],
            "Newspaper": [newspaper]
        })

        prediction = model.predict(input_data)
        st.success(f"Predicted Sales: {prediction[0]:.2f}")
    else:
        st.error("Model is not loaded. Cannot perform prediction.")