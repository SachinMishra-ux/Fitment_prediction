import streamlit as st
import pandas as pd
import datetime
from new_utils import (
    load_model, get_feature_names, transform_input, predict_with_ci,
    explain_prediction, plot_top_shap_features, show_shap_table, show_prediction_breakdown,
    get_benchmark_row, compare_with_benchmark, render_salary_recommendation
)

# ---------------------------
# Load Model & Benchmark Data
# ---------------------------
model_path = r"C:\Users\smm931389\Desktop\Fitment_Decision\model\hiring_model_v1.joblib"
benchmark_path = r"C:\Users\smm931389\Desktop\Fitment_Decision\new_data\Embedded Grid- Remade 29th July.xlsx"

preprocessor, rf_model = load_model(model_path)
feature_names = get_feature_names(preprocessor)

benchmark_df = pd.read_excel(benchmark_path)
benchmark_df.columns = benchmark_df.columns.str.strip()  # Clean col names

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("💼 Offered CTC Prediction with Market Comparison & Explainability")

# ---------------------------
# Candidate Input Form
# ---------------------------
st.header("📝 Enter Candidate Details")
date_of_joining = st.date_input("Date of Joining", value=datetime.date.today())
long_open_position_date = st.date_input("Long Open Position Date", value=datetime.date(2022, 1, 1))
year_of_joining = date_of_joining.year
days_since_joining = (datetime.date.today() - date_of_joining).days
long_open_flag = "Yes" if (date_of_joining - long_open_position_date).days > 183 else "No"

qualification = st.selectbox("Highest Qualification", ['B.Tech','M.Tech','MBA','PhD','Other'])
total_exp = st.number_input("Total Experience (years)", 0.0, 30.0, 5.0, step=0.1)
relevant_exp = st.number_input("Relevant Experience (years)", 0.0, 30.0, 4.0, step=0.1)
employment_type = st.selectbox("Employment Type", ['Full Time','Contract','Internship'])
job_location = st.text_input("Job Location", 'Pune')
job_level = st.text_input("Job Level (Grade)", 'L2')
billability = 1 if st.selectbox("Billability", ['Yes','IP']) == 'Yes' else 0
skill_tag = st.selectbox("Skill Tagging", ['Niche','Super-niche','Generic'])
skill_family = st.text_input("Skill Family", 'Advanced Embedded')
skill_name = st.text_input("Skill Name", 'Embedded Architect')
current_ctc = st.number_input("Current CTC (₹)", 1000000.0, 5000000.0, 2000000.0, step=50000.0)
offer_in_hand = st.number_input("Offer in Hand (₹)", 0.0, 5000000.0, 0.0, step=50000.0)
project_role = st.text_input("Project Role", 'Sr. Embedded Engineer')
client = st.text_input("Client", 'Volvo')

input_dict = {
    'Highest Qualification': qualification,
    'Total Experience (In Years)': total_exp,
    'Relevant Experience (In Years)': relevant_exp,
    'Employment Type': employment_type,
    'Job Location': job_location,
    'Job Level (Grade)': job_level,
    'Long Open Position': long_open_flag,
    'Billability': billability,
    'Skill Tagging': skill_tag,
    'Skill Family': skill_family,
    'Skill Name': skill_name,
    "Candidate's Current CTC (Pre TTL Last CTC)": current_ctc,
    'Offer in Hand': offer_in_hand,
    'Project Role': project_role,
    'Client': client,
    'Year of Joining': year_of_joining,
    'Days Since Joining': days_since_joining
}

input_df = pd.DataFrame([input_dict])

# ---------------------------
# Prediction & Market Comparison
# ---------------------------
if st.button("Predict Offered CTC"):
    try:
        # Prediction
        X_transformed = transform_input(preprocessor, input_df)
        y_pred, lower_ci, upper_ci = predict_with_ci(rf_model, X_transformed)

        st.success(f"💰 Predicted Offered CTC: ₹{int(y_pred):,}")
        st.info(f"📉 95% Confidence Interval: ₹{int(lower_ci):,} - ₹{int(upper_ci):,}")

        # SHAP Explainability
        shap_values, base_value, feature_names = explain_prediction(rf_model, X_transformed, feature_names)
        st.subheader("🔍 Top Feature Impacts (SHAP Explanation)")
        plot_top_shap_features(shap_values, feature_names)
        show_shap_table(shap_values, feature_names)
        show_prediction_breakdown(base_value, shap_values, y_pred)

        # Benchmark Matching
        benchmark_row = get_benchmark_row(benchmark_df, qualification, total_exp, skill_tag)
        if benchmark_row is not None:
            market_data = compare_with_benchmark(y_pred, benchmark_row)
            render_salary_recommendation(y_pred, market_data)
        else:
            st.warning("⚠ No matching benchmark found for given inputs.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
