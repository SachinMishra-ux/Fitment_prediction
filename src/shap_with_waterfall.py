import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.sparse import issparse

# === Load model and components ===
model = joblib.load(r"C:\Users\smm931389\Desktop\Fitment_Decision\model\offered_ctc_model.joblib")
preprocessor = model.named_steps['preprocessor']
rf_model = model.named_steps['regressor']

# === Get feature names after preprocessing ===
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

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("💼 Offered CTC Prediction with SHAP Explainability")

# === Input Form ===
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
    'Approvals': st.selectbox("Approvals", ['No Approval','Local Approval','Global Approval']),
    'Client': st.text_input("Client", 'Volvo'),
    'Year of Joining': year_of_joining,
    'Days Since Joining': days_since_joining
}

input_df = pd.DataFrame([input_dict])

# === Prediction and SHAP Visualization ===
if st.button("Predict Offered CTC"):
    try:
        # Preprocess input
        X_transformed = preprocessor.transform(input_df)
        if issparse(X_transformed):
            X_transformed = X_transformed.toarray()

        # Prediction from RF trees (for CI)
        preds = np.array([tree.predict(X_transformed)[0] for tree in rf_model.estimators_])
        y_pred = np.mean(preds)
        lower_ci = np.percentile(preds, 2.5)
        upper_ci = np.percentile(preds, 97.5)

        st.success(f"💰 Predicted Offered CTC: ₹{int(y_pred):,}")
        st.info(f"📉 95% Confidence Interval: ₹{int(lower_ci):,} - ₹{int(upper_ci):,}")

        # SHAP Explanation
        st.subheader("🔍 Top 10 Feature Impacts (SHAP)")
        explainer = shap.TreeExplainer(rf_model)
        X_df = pd.DataFrame(X_transformed, columns=feature_names)
        shap_values = explainer.shap_values(X_df)[0]

        # Top 10 SHAP
        abs_vals = np.abs(shap_values)
        top_indices = np.argsort(abs_vals)[-10:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_shap_vals = shap_values[top_indices]

        # Bar Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#d62728' if v < 0 else '#1f77b4' for v in top_shap_vals]
        ax.barh(range(len(top_features)), top_shap_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([top_features[i] for i in range(len(top_features)-1, -1, -1)])
        ax.set_xlabel("SHAP Value")
        ax.set_title("Top 10 SHAP Feature Contributions")
        st.pyplot(fig)

        # SHAP Table
        st.subheader("📊 SHAP Contribution Table")
        shap_table = pd.DataFrame({
            'Feature': top_features,
            'SHAP Value': [f"₹{int(v):,}" for v in top_shap_vals]
        })
        st.dataframe(shap_table)

        # Waterfall Plot
        st.subheader("📉 SHAP Waterfall Plot")
        base_value = explainer.expected_value
        contribution_sum = np.sum(shap_values)
        final_prediction = base_value + contribution_sum

        waterfall_features = top_features + ['Remaining Features']
        waterfall_values = list(top_shap_vals) + [contribution_sum - np.sum(top_shap_vals)]
        fig_waterfall = go.Figure(go.Waterfall(
            name="CTC Breakdown",
            orientation="v",
            measure=["relative"] * len(waterfall_values) + ["total"],
            x=waterfall_features + ["Prediction"],
            y=waterfall_values + [base_value + contribution_sum],
            base=base_value,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig_waterfall.update_layout(
            title="💧 SHAP Waterfall: Base CTC → Adjusted by Feature Effects → Final Prediction",
            showlegend=False,
            height=600
        )
        st.plotly_chart(fig_waterfall)

        # Explanation
        st.markdown(f"""
        **Prediction Breakdown**
        - Model Base Value (mean CTC): ₹{int(base_value):,}
        - Sum of SHAP contributions: ₹{int(contribution_sum):,}
        - Final Predicted Offered CTC: ₹{int(final_prediction):,}
        """)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
