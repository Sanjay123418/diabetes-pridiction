import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🩺")

# Set Background Color and Button Style for Medical Theme
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to right, #E6F7F1, #C1E8E0);
            color: #2E3A47;
        }
        .stButton>button {
            display: block;
            margin: 20px auto;
            background-color: #4CAF50;
            color: white;
            font-size: 120%;
            padding: 12px 24px;
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
            border: none;
            cursor: pointer;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45A049;
            transform: scale(1.05);
        }
        .result-container {
            background-color: #4CAF50;
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-top: 15px;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
        }
        input {
            border-radius: 5px;
            padding: 8px;
            border: 1px solid #ccc;
            font-size: 100%;
        }
        h1 {
            font-family: 'Helvetica Neue', sans-serif;
            color: #2E3A47;
            text-align: center;
        }
        h3 {
            font-family: 'Arial', sans-serif;
            color: #2E3A47;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model
diabetes_model_path = "daibetes_model.sav"
try:
    with open(diabetes_model_path, 'rb') as model_file:
        diabetes_model = pickle.load(model_file)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

st.title('🩺 Diabetes Prediction using Machine Learning 🩺')

# Layout for input fields
col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies', value="0")

with col2:
    Glucose = st.text_input('Glucose Level', value="0")

with col3:
    BloodPressure = st.text_input('Blood Pressure value', value="0")

with col1:
    SkinThickness = st.text_input('Skin Thickness value', value="0")

with col2:
    Insulin = st.text_input('Insulin Level', value="0")

with col3:
    BMI = st.text_input('BMI value', value="0")

with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', value="0")

with col2:
    Age = st.text_input('Age of the Person', value="0")

# Check Model Accuracy
if st.button('Show Model Accuracy'):
    try:
        test_data = pd.read_csv(r"D:\workshop 2\daibetes_model.sav")
        x_test = test_data.drop(columns=["Outcome"])
        y_test = test_data["Outcome"]
        y_pred = diabetes_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model Accuracy: {accuracy * 100:.2f}%")
    except Exception as e:
        st.error(f"Error calculating accuracy: {e}")

# Prediction Button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button('🩺 Diabetes Test Result 🩺'):
    try:
        # Convert all inputs to float
        user_input = np.array([
            float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
            float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)
        ]).reshape(1, -1)  # Ensure it's a 2D array

        # Make prediction
        diab_prediction = diabetes_model.predict(user_input)

        # Display result
        if diab_prediction[0] == 1:
            diab_diagnosis = '⚠️ The person is diabetic ⚠️'
        else:
            diab_diagnosis = '✅ The person is not diabetic ✅'

        st.markdown(f"<div class='result-container'>{diab_diagnosis}</div>", unsafe_allow_html=True)

    except ValueError:
        st.error("Please enter valid numerical values for all fields.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
st.markdown("</div>", unsafe_allow_html=True)
