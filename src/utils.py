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
    total_contrib = np.sum(shap_values[0])
    st.subheader("🧾 Prediction Breakdown")
    st.markdown(f"- **Model Base Value (mean CTC):** ₹{int(base_value):,}")
    st.markdown(f"- **Sum of SHAP contributions:** ₹{int(total_contrib):,}")
    st.markdown(f"- **Final Predicted Offered CTC:** ₹{int(prediction):,}")

def explain_prediction(model, X_transformed, feature_names):
    explainer = shap.TreeExplainer(model)
    if X_transformed.shape[1] != len(feature_names):
        feature_names = feature_names[:X_transformed.shape[1]]
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    shap_values = explainer.shap_values(X_df)
    base_value = explainer.expected_value
    return shap_values, base_value, feature_names


def plot_global_shap(model, preprocessor, full_df, max_display=15):
    """
    Generate global SHAP summary plots for overall model explainability.
    
    Args:
        model: Trained regression model (e.g., RandomForestRegressor)
        preprocessor: Fitted preprocessor (ColumnTransformer)
        full_df: Pandas DataFrame of the original (untransformed) dataset
        max_display: Number of top features to display in the summary plots
    """
    try:
        # Transform dataset
        X_transformed = transform_input(preprocessor, full_df)
        feature_names = get_feature_names(preprocessor)
        X_df = pd.DataFrame(X_transformed, columns=feature_names)

        # Fit SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)

        # Plot beeswarm summary (scatter)
        st.subheader("🌍 Global Feature Importance (SHAP Summary)")
        fig_summary = shap.summary_plot(shap_values, X_df, plot_type="dot", max_display=max_display, show=False)
        st.pyplot(bbox_inches='tight')

        # Plot bar chart summary
        st.subheader("📊 Mean Absolute SHAP Values (Overall Importance)")
        fig_bar = shap.summary_plot(shap_values, X_df, plot_type="bar", max_display=max_display, show=False)
        st.pyplot(bbox_inches='tight')

    except Exception as e:
        st.error(f"❌ Failed to generate global SHAP plots: {str(e)}")
