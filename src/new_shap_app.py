import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import shap
import matplotlib.pyplot as plt
from scipy.sparse import issparse

# === Load model pipeline ===
model = joblib.load(r"C:\Users\smm931389\Desktop\Fitment_Decision\model\offered_ctc_model.joblib")
preprocessor = model.named_steps['preprocessor']
rf_model = model.named_steps['regressor']

# === Utility: Extract feature names after transformation ===
def get_feature_names(column_transformer):
    output_features = []
    for name, trans, cols in column_transformer.transformers_:
        if hasattr(trans, 'get_feature_names_out'):
            names = trans.get_feature_names_out(cols)
        elif trans == 'passthrough':
            names = cols
        else:
            names = [f"{name}_{col}" for col in cols]
        output_features.extend(names)
    return output_features

feature_names = get_feature_names(preprocessor)

# === SHAP Plot ===
def plot_top_shap_features(shap_values, feature_names, top_k=10):
    vals = shap_values[0]
    abs_vals = np.abs(vals)
    top_indices = np.argsort(abs_vals)[-top_k:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_shap_vals = vals[top_indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#d62728' if val < 0 else '#1f77b4' for val in top_shap_vals]
    ax.barh(range(len(top_features)), top_shap_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("SHAP Value")
    ax.set_title("Top 10 Features Impacting the Prediction")
    st.pyplot(fig)

# === SHAP Table ===
def show_shap_table(shap_values, feature_names, top_k=10):
    vals = shap_values[0]
    abs_vals = np.abs(vals)
    top_indices = np.argsort(abs_vals)[-top_k:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_vals = vals[top_indices]

    df = pd.DataFrame({
        "Feature": top_features,
        "SHAP Value": top_vals
    })
    st.subheader("📊 SHAP Contribution Table")
    st.dataframe(df.style.format({"SHAP Value": "₹{:,.0f}"}))

# === Breakdown Explanation ===
def show_prediction_breakdown(base_value, shap_values, prediction):
    total_contrib = np.sum(shap_values[0])
    st.subheader("🧾 Prediction Breakdown")
    st.markdown(f"- **Model Base Value (mean CTC):** ₹{int(base_value):,}")
    st.markdown(f"- **Sum of SHAP contributions:** ₹{int(total_contrib):,}")
    st.markdown(f"- **Final Predicted Offered CTC:** ₹{int(prediction):,}")

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("💼 Offered CTC Prediction with Local Explainability")

# Input form
st.header("📝 Enter Candidate Details")
date_of_joining = st.date_input("Date of Joining", value=datetime.date.today())
year_of_joining = date_of_joining.year
days_since_joining = (datetime.date.today() - date_of_joining).days

input_dict = {
    'Highest Qualification': st.selectbox("Highest Qualification", ['B.Tech','M.Tech','MBA','PhD','Other']),
    'Total Experience': st.number_input("Total Experience (years)", 0.0, 30.0, 5.0, step=0.1),
    'Relevant Experience': st.number_input("Relevant Experience (years)", 0.0, 30.0, 4.0, step=0.1),
    'Employment Type': st.selectbox("Employment Type", ['Full Time','Contract','Internship']),
    'Job Location': st.text_input("Job Location", 'Pune'),
    'Job Level (Grade)': st.text_input("Job Level (Grade)", 'L2'),
    'Billability': 1 if st.selectbox("Billability", ['Yes','IP']) == 'Yes' else 0,
    'Skill Tagging': st.selectbox("Skill Tagging", ['Niche','Super-niche','Generic']),
    'Skill Family': st.text_input("Skill Family", 'Advanced Embedded'),
    'Skill Name': st.text_input("Skill Name", 'Embedded Architect'),
    "Candidate's Current CTC (Pre TTL Last CTC)": st.number_input("Current CTC (₹)", 1000000.0, 5000000.0, 2000000.0, step=50000.0),
    'Offer in Hand': st.number_input("Offer in Hand (₹)", 0.0, 5000000.0, 0.0, step=50000.0),
    'Project Role': st.text_input("Project Role", 'Sr. Embedded Engineer'),
    'Approvals': st.selectbox("Approvals", ['No Approval','Local Approval','Central Approval']),
    'Client': st.text_input("Client", 'Volvo'),
    'Year of Joining': year_of_joining,
    'Days Since Joining': days_since_joining
}

input_df = pd.DataFrame([input_dict])

# === Prediction and Explanation ===
if st.button("Predict Offered CTC"):
    try:
        X_transformed = preprocessor.transform(input_df)

        # Convert to dense if needed
        if issparse(X_transformed):
            X_transformed = X_transformed.toarray()

        # Predict using all trees for CI
        preds = np.array([tree.predict(X_transformed)[0] for tree in rf_model.estimators_])
        y_pred = np.mean(preds)
        lower_ci = np.percentile(preds, 2.5)
        upper_ci = np.percentile(preds, 97.5)

        st.success(f"💰 Predicted Offered CTC: ₹{int(y_pred):,}")
        st.info(f"📉 95% Confidence Interval: ₹{int(lower_ci):,} - ₹{int(upper_ci):,}")

        # SHAP explainability
        explainer = shap.TreeExplainer(rf_model)
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
        shap_values = explainer.shap_values(X_transformed_df)
        base_value = explainer.expected_value

        # Plot & table
        st.subheader("🔍 Top Feature Impacts (SHAP Explanation)")
        plot_top_shap_features(shap_values, feature_names)
        show_shap_table(shap_values, feature_names)
        show_prediction_breakdown(base_value, shap_values, y_pred)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
