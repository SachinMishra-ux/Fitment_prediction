import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import io


def compute_top_shap_features(shap_values, feature_names, top_k=10):
    """Return top K SHAP features and values (no plotting)."""
    vals = shap_values[0]
    abs_vals = np.abs(vals)
    top_indices = np.argsort(abs_vals)[-top_k:][::-1]
    return {
        "features": [feature_names[i] for i in top_indices],
        "values": vals[top_indices]
    }

def generate_shap_plot(shap_values, feature_names, top_k=10):
    """Generate SHAP feature importance plot and return fig."""
    top_data = compute_top_shap_features(shap_values, feature_names, top_k)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#d62728' if val < 0 else '#1f77b4' for val in top_data["values"]]
    ax.barh(range(len(top_data["features"])), top_data["values"][::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_data["features"])))
    ax.set_yticklabels(top_data["features"][::-1])
    ax.set_xlabel("SHAP Value")
    ax.set_title(f"Top {top_k} Features Impacting the Prediction")
    return fig

def generate_shap_plot_buffer(shap_values, feature_names, top_k=10):
    """Render SHAP bar plot to an in-memory buffer."""
    fig = generate_shap_plot(shap_values, feature_names, top_k)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def get_shap_table(shap_values, feature_names, top_k=10):
    """Return a pandas DataFrame of SHAP values."""
    top_data = compute_top_shap_features(shap_values, feature_names, top_k)
    return pd.DataFrame({
        "Feature": top_data["features"],
        "SHAP Value": top_data["values"]
    })

def get_prediction_breakdown(base_value, shap_values, prediction):
    """Return breakdown components of prediction."""
    total_contrib = np.sum(shap_values)
    residual = prediction - base_value
    return {
        "base_value": base_value,
        "total_contrib": total_contrib,
        "final_prediction": prediction,
        "residual": residual
    }

def explain_prediction(model, X_transformed, feature_names):
    """Run SHAP explanation and return raw values."""
    explainer = shap.TreeExplainer(model)
    if X_transformed.shape[1] != len(feature_names):
        feature_names = feature_names[:X_transformed.shape[1]]
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    shap_values = explainer.shap_values(X_df)
    base_value = explainer.expected_value
    return shap_values, base_value, feature_names


def make_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list)):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif pd.isna(obj):
        return None
    else:
        return obj
