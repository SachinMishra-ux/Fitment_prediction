import pandas as pd
from model_utils import *
from shap_utils import *

# Example candidate input (replace with your real values)
input_data = {
    'Highest Qualification': ['B.Tech'],
    'Total Experience (In Years)': [6.0],
    'Relevant Experience (In Years)': [5.0],
    'Employment Type': ['Full Time'],
    'Job Location': ['Pune'],
    'Job Level (Grade)': ['L2'],
    'Long Open Position': ['Yes'],
    'Billability': [1],
    'Skill Tagging': ['Super-niche'],
    'Skill Family': ['Advanced Embedded'],
    'Skill Name': ['Embedded Architect'],
    "Candidate's Current CTC (Pre TTL Last CTC)": [2000000],
    'Offer in Hand': [0],
    'Project Role': ['Sr. Embedded Engineer'],
    'Client': ['Volvo'],
    'Year of Joining': [2024],
    'Days Since Joining': [30]
}

input_df = pd.DataFrame(input_data)

# Load model
preprocessor, rf_model = load_model(r"C:\Users\smm931389\Desktop\Fitment_Decision\model\hiring_model_v1.joblib")

# Transform input
X_transformed = transform_input(preprocessor, input_df)

# Predict
pred_data = predict_with_ci(rf_model, X_transformed)
print("Predicted Mean:", pred_data["mean"])
print("95% CI:", pred_data["lower_ci"], "-", pred_data["upper_ci"])

# Explain
shap_vals, base_val, feat_names = explain_prediction(
    rf_model,
    X_transformed,
    get_feature_names(preprocessor)
)

# Get breakdown
breakdown = get_prediction_breakdown(base_val, shap_vals.sum(), pred_data["mean"])
print("\nBreakdown:", breakdown)

# Get SHAP table
shap_table = get_shap_table(shap_vals, feat_names)
print("\nSHAP Table:\n", shap_table)

# Get SHAP plot
fig = generate_shap_plot(shap_vals, feat_names)
fig.show()
