import streamlit as st
import pandas as pd
import datetime
from utils import (
    load_model, get_feature_names, transform_input, predict_with_ci
)

# === Load model and preprocessor ===
model_path = r"C:\Users\smm931389\Desktop\Fitment_Decision\experiment\hiring_model_v1.joblib"
preprocessor, rf_model = load_model(model_path)
feature_names = get_feature_names(preprocessor)

st.set_page_config(layout="wide")
st.title("💼 Offered CTC Prediction with Local Explainability")

# Input form
st.header("📝 Enter Candidate Details")
date_of_joining = st.date_input("Date of Joining", value=datetime.date.today())
long_open_position_date = st.date_input("Long Open Position Date", value=datetime.date(2022, 1, 1))
year_of_joining = date_of_joining.year
days_since_joining = (datetime.date.today() - date_of_joining).days
long_open_flag = "Yes" if (date_of_joining - long_open_position_date).days > 183 else "No"

input_dict = {
    'Highest Qualification': st.selectbox("Highest Qualification", ['B.Tech','M.Tech','MBA','PhD','Other']),
    'Total Experience (In Years)': st.number_input("Total Experience (years)", 0.0, 30.0, 5.0, step=0.1),
    'Relevant Experience (In Years)': st.number_input("Relevant Experience (years)", 0.0, 30.0, 4.0, step=0.1),
    'Employment Type': st.selectbox("Employment Type", ['Full Time','Contract','Internship']),
    'Job Location': st.text_input("Job Location", 'Pune'),
    'Job Level (Grade)': st.text_input("Job Level (Grade)", 'L2'),
    'Long Open Position': long_open_flag,
    'Billability': 1 if st.selectbox("Billability", ['Yes','IP']) == 'Yes' else 0,
    'Skill Tagging': st.selectbox("Skill Tagging", ['Niche','Super-niche','Generic']),
    'Skill Family': st.text_input("Skill Family", 'Advanced Embedded'),
    'Skill Name': st.text_input("Skill Name", 'Embedded Architect'),
    "Candidate's Current CTC (Pre TTL Last CTC)": st.number_input("Current CTC (₹)", 1000000.0, 5000000.0, 2000000.0, step=50000.0),
    'Offer in Hand': st.number_input("Offer in Hand (₹)", 0.0, 5000000.0, 0.0, step=50000.0),
    'Project Role': st.text_input("Project Role", 'Sr. Embedded Engineer'),
    'Client': st.text_input("Client", 'Volvo'),
    'Year of Joining': year_of_joining,
    'Days Since Joining': days_since_joining
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Offered CTC"):
    try:
        X_transformed = transform_input(preprocessor, input_df)
        y_pred, lower_ci, upper_ci = predict_with_ci(rf_model, X_transformed)

        st.success(f"💰 Predicted Offered CTC: ₹{int(y_pred):,}")
        st.info(f"📉 95% Confidence Interval: ₹{int(lower_ci):,} - ₹{int(upper_ci):,}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")