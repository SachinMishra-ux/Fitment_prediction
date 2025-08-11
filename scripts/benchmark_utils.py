import pandas as pd

def get_experience_bracket(total_exp):
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
    exp_bracket = get_experience_bracket(total_exp)
    if not exp_bracket:
        return None

    benchmark_df.columns = benchmark_df.columns.str.strip()

    if qualification.lower() in ["b.tech", "be", "b.e.", "b.e./b.tech"]:
        grade_col = "Grade with B.E./ B.Tech"
    else:
        grade_col = "Grade with M.E./ M.Tech"

    df_filtered = benchmark_df[
        (benchmark_df["Experience Bracket"].str.strip() == exp_bracket) &
        (benchmark_df["Skill Tagging"].str.strip().str.lower() == skill_tag.lower())
    ]

    if df_filtered.empty:
        return None

    return df_filtered.iloc[0]

def compare_with_benchmark(predicted_ctc, benchmark_row):
    p25 = benchmark_row["P25"]
    p50 = benchmark_row["P50"]
    p75 = benchmark_row["P75"]
    p90 = benchmark_row["P90"]
    max_val = benchmark_row["Max"]

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

    deviation_pct = ((predicted_ctc - p50) / p50) * 100

    return {
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "max": max_val,
        "position": position,
        "market_deviation_pct": deviation_pct
    }


def decide_fitment(suggested_ctc, market_data, grade, internal_dev):
    deviation = float(market_data["market_deviation_pct"])

    # Fitment decision: Accept only if suggested_ctc <= P50
    if suggested_ctc <= market_data["p50"]:
        decision = "Accept"
        rationale = f"CTC is less than or equal to P50 ({int(market_data['p50']):,}), hence accepted."
    else:
        decision = "Review"
        rationale = f"CTC is greater than P50 ({int(market_data['p50']):,}), hence needs review."

    # CHRO exception logic
    is_exception = suggested_ctc > market_data["p90"]
    chro_exception = "Yes" if is_exception else "No"

    # Market range string
    market_range = f"{int(market_data['p50']):,}–{int(market_data['p90']):,}"

    # Alternative suggestion logic
    if is_exception:
        alt_suggestion = f"Predicted CTC exceeds P90. Consider CHRO approval or adjusting grade from {grade}."
    else:
        alt_suggestion = "Proceed with negotiation"

    return {
        "fitment_decision": decision,
        "grade_recommendation": grade,
        "suggested_ctc": float(suggested_ctc),
        "internal_deviation_percent": float(internal_dev) if internal_dev is not None else None,
        "market_deviation_percent": deviation,
        "market_range(P50-P90)": market_range,
        "chro_exception": chro_exception,
        "alternative_suggestions": alt_suggestion,
        "rationale": rationale
    }



