import joblib
import pandas as pd

# Load user input
# input_df = pd.DataFrame(...) <- from Streamlit form

# Load saved scaler and feature list
scaler = joblib.load('rscaler.pkl')
trained_features = joblib.load('rmodel.pkl')

# Ensure all required columns exist in input_df
for col in trained_features:
    if col not in input_df.columns:
        input_df[col] = 0  # or a default value

# Reorder columns to match training
input_df = input_df[trained_features]

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
