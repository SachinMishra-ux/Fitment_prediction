import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import streamlit as st

def load_model(model_path):
    model = joblib.load(model_path)
    return model.named_steps['preprocessor'], model.named_steps['regressor']

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

def transform_input(preprocessor, input_df):
    X_transformed = preprocessor.transform(input_df)
    if issparse(X_transformed):
        X_transformed = X_transformed.toarray()
    return X_transformed

def predict_with_ci(model, X_transformed):
    preds = np.array([tree.predict(X_transformed)[0] for tree in model.estimators_])
    return np.mean(preds), np.percentile(preds, 2.5), np.percentile(preds, 97.5)

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

def show_shap_table(shap_values, feature_names, top_k=10):
    vals = shap_values[0]
    abs_vals = np.abs(vals)
    top_indices = np.argsort(abs_vals)[-top_k:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_vals = vals[top_indices]

    df = pd.DataFrame({"Feature": top_features, "SHAP Value": top_vals})
    st.subheader("📊 SHAP Contribution Table")
    st.dataframe(df.style.format({"SHAP Value": "₹{:,.0f}"}))

def show_prediction_breakdown(base_value, shap_values, prediction):

    total_contrib = np.sum(shap_values)
    residual = prediction - base_value

    st.write("### 📊 Prediction Breakdown")
    st.write(f"🔹 **Base Value (Model Bias):** ₹{int(base_value):,}")
    st.write(f"🔹 **Sum of SHAP Contributions:** ₹{int(total_contrib):,}")
    st.write(f"🔹 **Final Prediction:** ₹{int(prediction):,}")
    st.write(f"🔹 **Residual (Prediction - Base):** ₹{int(residual):,}")

def explain_prediction(model, X_transformed, feature_names):
    explainer = shap.TreeExplainer(model)
    if X_transformed.shape[1] != len(feature_names):
        feature_names = feature_names[:X_transformed.shape[1]]
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    shap_values = explainer.shap_values(X_df)
    base_value = explainer.expected_value
    return shap_values, base_value, feature_names




# ---------------------------
# Benchmark Matching
# ---------------------------
def get_experience_bracket(total_exp):
    """Map total experience to benchmark bracket."""
    brackets = [
        (0, 2, "0-2 Year"),
        (2, 4, "2-4 Years"),
        (4, 6, "4-6 Years"),
        (6, 8, "6-8 Years"),
        (8, 10, "8-10 Years"),
        (10, 12, "10-12 Years"),
        (12, 15, "12-15 Years"),
        (15, 20, "15-20 Years"),
        (20, 25, "20-25 Years"),
    ]
    for low, high, label in brackets:
        if low <= total_exp < high:
            return label
    return None


def get_benchmark_row(benchmark_df, qualification, total_exp, skill_tag):
    """Return matching benchmark row based on qualification, experience & skill tagging."""
    exp_bracket = get_experience_bracket(total_exp)
    if not exp_bracket:
        return None

    # Clean columns
    benchmark_df.columns = benchmark_df.columns.str.strip()

    if qualification.lower() in ["b.tech", "be", "b.e.", "b.e./b.tech"]:
        grade_col = "Grade with B.E./ B.Tech"
    else:
        grade_col = "Grade with M.E./ M.Tech"

    # Match rows
    df_filtered = benchmark_df[
        (benchmark_df["Experience Bracket"].str.strip() == exp_bracket) &
        (benchmark_df["Skill Tagging"].str.strip().str.lower() == skill_tag.lower())
    ]

    if df_filtered.empty:
        return None

    return df_filtered.iloc[0]


# ---------------------------
# Market Comparison Logic
# ---------------------------
def compare_with_benchmark(predicted_ctc, benchmark_row):
    """Compare predicted CTC with market benchmark."""
    p25 = benchmark_row["P25"]
    p50 = benchmark_row["P50"]
    p75 = benchmark_row["P75"]
    p90 = benchmark_row["P90"]
    max_val = benchmark_row["Max"]

    # Determine position
    if predicted_ctc < p25:
        position = "Below P25"
    elif p25 <= predicted_ctc < p50:
        position = "P25–P50"
    elif p50 <= predicted_ctc < p75:
        position = "P50–P75"
    elif p75 <= predicted_ctc < p90:
        position = "P75–P90"
    elif p90 <= predicted_ctc <= max_val:
        position = "P90–Max"
    else:
        position = "Above Max"

    # Market deviation %
    median_val = p50
    deviation_pct = ((predicted_ctc - median_val) / median_val) * 100

    return {
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "max": max_val,
        "position": position,
        "market_deviation_pct": deviation_pct
    }


# ---------------------------
# UI Rendering
# ---------------------------
def render_salary_recommendation(predicted_ctc, benchmark_data):
    """Render salary recommendation UI."""
    position = benchmark_data["position"]
    deviation = benchmark_data["market_deviation_pct"]

    # Overall Recommendation Text
    if position in ["P50–P75", "P75–P90"]:
        decision = "Accept"
        summary_text = f"Candidate is recommended with predicted CTC ₹{predicted_ctc:,.0f} — within {position} of market."
    else:
        decision = "Review"
        summary_text = f"Candidate's predicted CTC ₹{predicted_ctc:,.0f} falls in {position} — review before approval."

    st.markdown("### 📌 Overall Recommendation")
    st.info(summary_text)

    # Salary Proposal
    st.markdown("### 💰 Salary Proposal")
    st.write(f"**Fitment Decision:** {decision}")
    st.write(f"**Suggested CTC:** ₹{predicted_ctc:,.0f}")
    st.write(f"**Market Range:** ₹{benchmark_data['p50']:,.0f} – ₹{benchmark_data['p90']:,.0f}")
    st.write(f"**Position:** {position}")
    st.write(f"**Market Deviation %:** {deviation:.2f}%")

    # Benchmark Table
    st.markdown("### 📊 Benchmark Data")
    st.table(pd.DataFrame({
        "P25": [benchmark_data['p25']],
        "P50": [benchmark_data['p50']],
        "P75": [benchmark_data['p75']],
        "P90": [benchmark_data['p90']],
        "Max": [benchmark_data['max']]
    }))
