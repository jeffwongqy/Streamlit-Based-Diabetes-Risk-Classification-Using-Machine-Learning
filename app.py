import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Provide the patient details below to predict diabetes risk.")  

model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", options = ['Male', 'Female'])
    age = st.number_input("Age", min_value = 0, max_value = 100, step = 1)
    hypertension = st.radio("Hypertension", options = ['Yes', 'No'])
    heart_disease = st.radio("Heart Disease", options = ['Yes', 'No'])

with col2:
    smoking_history = st.radio("Smoking History", options = ['Current', 'Ever', 'Former', 'Never', 'Not Current', 'No Info'])
    bmi = st.number_input("BMI", min_value = 0.0, max_value = 100.0, format = "%.2f")
    hba1c_level = st.number_input("HbA1c Level", min_value = 0.0, max_value = 20.0, format = "%.1f")
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value = 0, max_value = 500, step = 1)

if gender == "Female":
    gender = 0 
elif gender == "Male":
    gender = 1
    
if hypertension == "No":
    hypertension = 0
elif hypertension == "Yes":
    hypertension = 1
    
if heart_disease == "No":
    heart_disease = 0
elif heart_disease == "Yes":
    heart_disease = 1

if smoking_history == "Current":
    smoking_history = 0
elif smoking_history == "Ever":
    smoking_history = 1
elif smoking_history == "Former":
    smoking_history = 2
elif smoking_history == "Never":
    smoking_history = 3
elif smoking_history == "Not Current":
    smoking_history = 4
elif smoking_history == "No Info":
    smoking_history = 5


if st.button("Predict Diabetes"):
    numerical_features = np.array([[age, bmi, hba1c_level, blood_glucose_level]])
    scaled_numerical = scaler.transform(numerical_features)
    scaled_age, scaled_bmi, scaled_hba1c_level, scaled_blood_glucose_level = scaled_numerical[0]
    
    final_features = np.array([[gender, 
                                scaled_age, 
                                hypertension, 
                                heart_disease, 
                                smoking_history, 
                                scaled_bmi, 
                                scaled_hba1c_level, 
                                scaled_blood_glucose_level]])
    
    prediction = model.predict(final_features)
    
    st.divider()
    st.subheader("Result")
    
    if prediction == "1":
        st.error("Prediction: Positive for Diabetes!")
    else:
        st.success("Prediction: Negative for Diabetes!")
    