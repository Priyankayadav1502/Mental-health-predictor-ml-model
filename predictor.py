import streamlit as st
import joblib
import pandas as pd

# Load model components
model = joblib.load('rmodel.pkl')
scaler = joblib.load('rscaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')  # This must match features used during training

st.title("🧠 Mental Health Treatment Predictor")

# User input form
st.header("Enter Survey Information")

user_input = {
    'Age': st.slider('Age', 18, 100, 30),
    'Gender': st.selectbox('Gender', ['Male', 'Female', 'Other']),
    'self_employed': st.selectbox('Self Employed', ['Yes', 'No']),
    'family_history': st.selectbox('Family History of Mental Illness', ['Yes', 'No']),
    'work_interfere': st.selectbox('Mental Health Interferes With Work', ['Never', 'Rarely', 'Sometimes', 'Often']),
    'no_employees': st.selectbox('Company Size', ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']),
    'remote_work': st.selectbox('Remote Work', ['Yes', 'No']),
    'tech_company': st.selectbox('Tech Company', ['Yes', 'No']),
    'benefits': st.selectbox('Has Mental Health Benefits', ['Yes', 'No', "Don't know"]),
    'care_options': st.selectbox('Access to Care Options', ['Not sure', 'No', 'Yes']),
    'wellness_program': st.selectbox('Wellness Program', ['Yes', 'No', "Don't know"]),
    'seek_help': st.selectbox('Seek Help Resources', ['Yes', 'No', "Don't know"]),
    'anonymity': st.selectbox('Anonymity Provided', ['Yes', 'No', "Don't know"]),
    'leave': st.selectbox('Ease of Taking Mental Health Leave', ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"]),
    'mental_health_consequence': st.selectbox('Consequences of Disclosure', ['Yes', 'No', 'Maybe']),
    'phys_health_consequence': st.selectbox('Physical Health Disclosure Consequences', ['Yes', 'No', 'Maybe']),
    'coworkers': st.selectbox('Comfort Discussing with Coworkers', ['Yes', 'No', 'Some of them']),
    'supervisor': st.selectbox('Comfort Discussing with Supervisor', ['Yes', 'No', 'Some of them']),
    'mental_health_interview': st.selectbox('Disclose in Interview?', ['Yes', 'No', 'Maybe']),
    'phys_health_interview': st.selectbox('Disclose Physical Health in Interview?', ['Yes', 'No', 'Maybe']),
    'mental_vs_physical': st.selectbox('Mental vs Physical Health Perceived Importance', ['Yes', 'No', 'Don\'t know']),
    'obs_consequence': st.selectbox('Observed Consequences for Others', ['Yes', 'No']),
    'country': st.selectbox('Country', ['United States', 'Canada', 'United Kingdom', 'India', 'Germany']),
    'state': st.text_input('State (can be blank)', '')
}

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Encode categorical variables
for column in input_df.columns:
    if column in label_encoders:
        encoder = label_encoders[column]
        try:
            input_df[column] = encoder.transform(input_df[column])
        except:
            # Use most frequent value from training if unseen
            input_df[column] = encoder.transform([encoder.classes_[0]])

# Ensure all feature columns exist
missing_cols = [col for col in feature_columns if col not in input_df.columns]
for col in missing_cols:
    input_df[col] = 0

# Reorder columns to match training
input_df = input_df[feature_columns]

# Scale data
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict Treatment Need"):
    prediction = model.predict(input_scaled)[0]
    st.subheader("Prediction Result:")
    st.success("✅ Treatment may be needed." if prediction == 1 else "❌ No treatment needed.")
