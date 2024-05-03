import streamlit as st
from joblib import load
import numpy as np
import pandas as pd

# Load the model
knn = load('knn_model.joblib')
svm = load('svm_model.joblib')

def get_user_input():
    st.write("Please provide the following information:")
    age = st.number_input("Enter your Age", min_value=0, max_value=120)
    avg_glucose_level = st.number_input("Enter your Average Glucose Level")
    bmi = st.number_input("Enter your BMI")
    gender_Male = st.radio("Gender", options=["Male", "Female"]) == 'Male'
    hypertension_1 = st.radio("Do you have Hypertension?", options=["Yes", "No"]) == 'Yes'
    heart_disease_1 = st.radio("Do you have Heart Disease?", options=["Yes", "No"]) == 'Yes'
    ever_married_Yes = st.radio("Ever Married?", options=["Yes", "No"]) == 'Yes'
    
    work_type = st.radio("Type of work", options=["Private", "Never worked", "Self-Employed", "Children"])
    work_type_Never_worked = work_type == 'Never worked'
    work_type_Private = work_type == 'Private'
    work_type_Self_employed = work_type == 'Self-Employed'
    work_type_children = work_type == 'Children'
    
    Residence_type_Urban = st.radio("Are you an Urban resident?", options=["Yes", "No"]) == 'Yes'
    
    smoking_status = st.radio("Do you smoke?", options=["Never", "Formerly", "Smokes"])
    smoking_status_formerly_smoked = smoking_status == 'Formerly'
    smoking_status_never_smoked = smoking_status == 'Never'
    smoking_status_smokes = smoking_status == 'Smokes'
    
    # Return user input as a dictionary
    return {
        'age': age,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender_Male': gender_Male,
        'hypertension_1': hypertension_1,
        'heart_disease_1': heart_disease_1,
        'ever_married_Yes': ever_married_Yes,
        'work_type_Never_worked': work_type_Never_worked,
        'work_type_Private': work_type_Private,
        'work_type_Self-employed': work_type_Self_employed,
        'work_type_children': work_type_children,
        'Residence_type_Urban': Residence_type_Urban,
        'smoking_status_formerly smoked': smoking_status_formerly_smoked,
        'smoking_status_never smoked': smoking_status_never_smoked,
        'smoking_status_smokes': smoking_status_smokes
    }

# Get user input within Streamlit app
user_input = get_user_input()

# Ensure user input matches the expected features
expected_features = ['age', 'avg_glucose_level', 'bmi', 'gender_Male', 'hypertension_1', 
                     'heart_disease_1', 'ever_married_Yes', 'work_type_Never_worked', 
                     'work_type_Private', 'work_type_Self-employed', 'work_type_children', 
                     'Residence_type_Urban', 'smoking_status_formerly smoked', 
                     'smoking_status_never smoked', 'smoking_status_smokes']

# Check if all expected features are present in user input
missing_features = [feature for feature in expected_features if feature not in user_input]
if missing_features:
    st.write(f"Please provide the missing features: {', '.join(missing_features)}")
else:
    # Convert user input to a DataFrame with a single row
    df = pd.DataFrame([user_input])
    
    # Make prediction
    prediction1 = knn.predict(df)[0]
    prediction2 = svm.predict(df)[0]
    st.write("Predicted stroke using KNN:", prediction1)
    st.write("Predicted stroke using SVM:", prediction2)