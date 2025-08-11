from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from schema_validation_api import PredictionRequest
import pandas as pd
import numpy as np
import base64
from constants import BENCHMARK_PATH, MODEL_PATH
from model_utils import load_model, get_feature_names, transform_input, predict_with_ci

from shap_utils import (
    get_prediction_breakdown,
    explain_prediction,
    make_json_serializable,
    get_shap_table,
    generate_shap_plot_buffer
)

from benchmark_utils import get_benchmark_row, compare_with_benchmark, decide_fitment

# Initialize FastAPI
app = FastAPI(
    title="Offered CTC Prediction API",
    description="Predicts Offered CTC and explains predictions using SHAP",
    version="1.0",
)

# === Load Model and Preprocessor ===

preprocessor, rf_model = load_model(MODEL_PATH)

# === Load Benchmark Data ===

benchmark_df = pd.read_excel(BENCHMARK_PATH)
benchmark_df.columns = benchmark_df.columns.str.strip()


@app.post("/predict")
def predict_ctc(request: PredictionRequest):
    # --- Convert Input to DataFrame ---
    input_dict = {
        "Highest Qualification": [request.Highest_Qualification],
        "Total Experience (In Years)": [request.Total_Experience],
        "Relevant Experience (In Years)": [request.Relevant_Experience],
        "Employment Type": [request.Employment_Type],
        "Job Location": [request.Job_Location],
        "Job Level (Grade)": [request.Job_Level_Grade],
        "Long Open Position": [request.Long_Open_Position],
        "Billability": [request.Billability],
        "Skill Tagging": [request.Skill_Tagging],
        "Skill Family": [request.Skill_Family],
        "Skill Name": [request.Skill_Name],
        "Candidate's Current CTC (Pre TTL Last CTC)": [request.Current_CTC],
        "Offer in Hand": [request.Offer_in_Hand],
        "Project Role": [request.Project_Role],
        "Client": [request.Client],
        "Year of Joining": [request.Year_of_Joining],
        "Days Since Joining": [request.Days_Since_Joining],
    }
    input_df = pd.DataFrame(input_dict)

    # --- Transform & Predict ---
    X_trans = transform_input(preprocessor, input_df)
    pred = predict_with_ci(rf_model, X_trans)
    suggested_ctc = float(pred["mean"])

    # --- Benchmark Fitment Logic ---
    benchmark_row = get_benchmark_row(
        benchmark_df,
        qualification=request.Highest_Qualification,
        total_exp=request.Total_Experience,
        skill_tag=request.Skill_Tagging,
    )

    # Optional: Calculate internal deviation if current CTC is provided
    internal_dev = (
        ((suggested_ctc - request.Current_CTC) / request.Current_CTC) * 100
        if request.Current_CTC and request.Current_CTC > 0
        else None
    )

    if benchmark_row is not None:
        market_data = compare_with_benchmark(suggested_ctc, benchmark_row)
        fitment_summary = decide_fitment(
            suggested_ctc, market_data, request.Job_Level_Grade, internal_dev
        )
    else:
        market_data = None
        fitment_summary = {
            "fitment_decision": "Insufficient Benchmark Data",
            "grade_recommendation": request.Job_Level_Grade,
            "suggested_ctc": suggested_ctc,
            "internal_deviation_percent": (
                float(internal_dev) if internal_dev is not None else None
            ),
            "market_deviation_percent": None,
            "market_range(P50-P90)": None,
            "chro_exception": "No",
            "alternative_suggestions": "Insufficient benchmark data to provide suggestion",
            "rationale": "Benchmark data unavailable for the given skill/experience bracket.",
        }

    return {
        "prediction": {
            "mean": suggested_ctc,
            "lower_ci": float(pred["lower_ci"]),
            "upper_ci": float(pred["upper_ci"]),
        },
        "fitment_summary": fitment_summary,
    }


@app.post("/shap_explain")
def shap_explain(request: PredictionRequest):
    # --- Convert Input to DataFrame ---
    input_dict = {
        "Highest Qualification": [request.Highest_Qualification],
        "Total Experience (In Years)": [request.Total_Experience],
        "Relevant Experience (In Years)": [request.Relevant_Experience],
        "Employment Type": [request.Employment_Type],
        "Job Location": [request.Job_Location],
        "Job Level (Grade)": [request.Job_Level_Grade],
        "Long Open Position": [request.Long_Open_Position],
        "Billability": [request.Billability],
        "Skill Tagging": [request.Skill_Tagging],
        "Skill Family": [request.Skill_Family],
        "Skill Name": [request.Skill_Name],
        "Candidate's Current CTC (Pre TTL Last CTC)": [request.Current_CTC],
        "Offer in Hand": [request.Offer_in_Hand],
        "Project Role": [request.Project_Role],
        "Client": [request.Client],
        "Year of Joining": [request.Year_of_Joining],
        "Days Since Joining": [request.Days_Since_Joining],
    }
    input_df = pd.DataFrame(input_dict)

    # --- Transform & SHAP Explanation ---
    X_trans = transform_input(preprocessor, input_df)
    shap_vals, base_val, feat_names = explain_prediction(
        rf_model, X_trans, get_feature_names(preprocessor)
    )

    # SHAP Image
    buf = generate_shap_plot_buffer(shap_vals, feat_names)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Breakdown
    pred = predict_with_ci(rf_model, X_trans)
    breakdown_dict = get_prediction_breakdown(base_val, shap_vals[0], pred["mean"])

    # SHAP Table
    shap_table = get_shap_table(shap_vals, feat_names)

    return {
        "shap_image": image_base64,
        "breakdown": make_json_serializable(breakdown_dict),
        "shap_table": [
            make_json_serializable(row)
            for row in shap_table.astype(object).to_dict(orient="records")
        ],
    }


# === Main for Local Dev ===
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_api:app", host="0.0.0.0", port=8000)
