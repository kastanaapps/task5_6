import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "report_task5_6.txt"
METRICS_PATH = OUTPUT_DIR / "metrics_task6.json"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load diabetes dataset from public URL."""
    return pd.read_csv(DATA_URL)


def detect_and_handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    In this dataset, zeros in clinical columns are medically implausible and treated as missing.
    We replace them with median values (robust to skew/outliers).
    """
    df_clean = df.copy()

    zero_as_missing_cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    missing_before = df_clean.isna().sum().to_dict()

    for col in zero_as_missing_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    for col in zero_as_missing_cols:
        median_value = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_value)

    return df_clean, missing_before


def cap_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Cap outliers using IQR rule (winsorization)."""
    result = df.copy()
    for col in columns:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        result[col] = result[col].clip(lower=lower, upper=upper)
    return result


def summary_statistics_by_outcome(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Outcome")["Age"].agg(["mean", "std", "count"]).rename(
        index={0: "Non-Diabetic", 1: "Diabetic"}
    )


def pairwise_correlations_sorted(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))

    result = pd.DataFrame(pairs, columns=["Feature_1", "Feature_2", "Correlation"])
    result["Abs_Correlation"] = result["Correlation"].abs()
    result = result.sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)
    return result


def answer_filtered_questions(df: pd.DataFrame) -> Dict[str, str]:
    answers = {}

    q1_data = df[(df["Pregnancies"] > 0) & (df["Age"] > 35)]
    answers["avg_insulin_pregnant_over_35"] = (
        f"{q1_data['Insulin'].mean():.2f}" if not q1_data.empty else "No data"
    )

    # The classic Pima dataset has no gender column and includes only female records.
    if "Gender" in df.columns:
        male_diabetic = df[(df["Gender"].str.lower().isin(["male", "m", "man"])) & (df["Outcome"] == 1)]
        answers["min_bmi_male_diabetic"] = (
            f"{male_diabetic['BMI'].min():.2f}" if not male_diabetic.empty else "No male diabetic records"
        )

        female_non_diabetic = df[
            (df["Gender"].str.lower().isin(["female", "f", "woman"])) & (df["Outcome"] == 0)
        ]
        if female_non_diabetic.empty:
            answers["bp_range_female_non_diabetic"] = "No female non-diabetic records"
        else:
            answers["bp_range_female_non_diabetic"] = (
                f"{female_non_diabetic['BloodPressure'].min():.2f} - "
                f"{female_non_diabetic['BloodPressure'].max():.2f}"
            )

        male_age_over_40 = df[(df["Gender"].str.lower().isin(["male", "m", "man"])) & (df["Age"] > 40)]
        if male_age_over_40.empty:
            answers["bp_range_male_over_40"] = "No male records over 40"
        else:
            answers["bp_range_male_over_40"] = (
                f"{male_age_over_40['BloodPressure'].min():.2f} - "
                f"{male_age_over_40['BloodPressure'].max():.2f}"
            )
    else:
        answers["min_bmi_male_diabetic"] = "Unavailable (dataset has no Gender column)"
        answers["bp_range_female_non_diabetic"] = (
            "Unavailable (dataset has no Gender column; dataset is female-only)"
        )
        answers["bp_range_male_over_40"] = "Unavailable (dataset has no Gender column)"

    return answers


def plot_task5_figures(df: pd.DataFrame) -> None:
    # 1) Boxplot: Age vs Outcome
    plt.figure(figsize=(7, 5))
    df.boxplot(column="Age", by="Outcome")
    plt.title("Age Distribution by Diabetes Outcome")
    plt.suptitle("")
    plt.xlabel("Outcome (0=No, 1=Yes)")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "boxplot_age_outcome.png", dpi=200)
    plt.close()

    # 2) Barchart: Diabetes rate by BMI category
    bins = [0, 18.5, 25, 30, 100]
    labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df_bmi = df.copy()
    df_bmi["BMI_Category"] = pd.cut(df_bmi["BMI"], bins=bins, labels=labels, include_lowest=True)
    bmi_outcome_rate = df_bmi.groupby("BMI_Category", observed=False)["Outcome"].mean() * 100
    plt.figure(figsize=(8, 5))
    bmi_outcome_rate.plot(kind="bar")
    plt.title("Diabetes Percentage by BMI Category")
    plt.xlabel("BMI Category")
    plt.ylabel("Diabetes %")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "barchart_bmi_vs_diabetes_rate.png", dpi=200)
    plt.close()

    # 3) Line-like trend: BMI vs Glucose by sorted BMI + rolling mean
    trend = df[["BMI", "Glucose"]].sort_values("BMI").reset_index(drop=True)
    trend["Glucose_Rolling"] = trend["Glucose"].rolling(window=25, min_periods=5).mean()
    plt.figure(figsize=(8, 5))
    plt.plot(trend["BMI"], trend["Glucose_Rolling"])
    plt.title("Trend Between BMI and Glucose (Rolling Mean)")
    plt.xlabel("BMI")
    plt.ylabel("Glucose (rolling mean)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "line_bmi_glucose_trend.png", dpi=200)
    plt.close()


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return x, y


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def expert_system_predict(x: pd.DataFrame) -> np.ndarray:
    """
    Simple rule base derived from domain-like thresholds.
    A record is predicted diabetic if at least 2 rules are true.
    """
    rule1 = x["Glucose"] >= 125
    rule2 = x["BMI"] >= 30
    rule3 = x["Age"] >= 45
    rule4 = x["DiabetesPedigreeFunction"] >= 0.6
    rule5 = x["Insulin"] >= 150

    score = rule1.astype(int) + rule2.astype(int) + rule3.astype(int) + rule4.astype(int) + rule5.astype(int)
    return (score >= 2).astype(int).values


def train_ml_models(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Dict[str, float]]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=9),
    }

    metrics = {}
    for name, model in models.items():
        if name in {"LogisticRegression", "KNN"}:
            model.fit(x_train_scaled, y_train)
            pred = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
        metrics[name] = evaluate_predictions(y_test, pred)

    return metrics


def plot_task6_comparison(metrics: Dict[str, Dict[str, float]]) -> None:
    df_metrics = pd.DataFrame(metrics).T
    plt.figure(figsize=(9, 5))
    df_metrics.plot(kind="bar")
    plt.title("Model Comparison (Accuracy / Precision / Recall)")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task6_models_comparison.png", dpi=200)
    plt.close()


def save_report(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    missing_before: Dict[str, int],
    age_stats: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    filtered_answers: Dict[str, str],
    expert_metrics: Dict[str, float],
    ml_metrics: Dict[str, Dict[str, float]],
) -> None:
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("=== Task 5 & 6 Diabetes Analysis Report ===\n\n")
        f.write("Part 1 - Data Loading & Exploration\n")
        f.write(f"- Raw shape: {raw_df.shape}\n")
        f.write(f"- Clean shape: {clean_df.shape}\n\n")

        f.write("First 5 rows:\n")
        f.write(raw_df.head().to_string(index=False))
        f.write("\n\n")

        f.write("Dataset info (dtypes):\n")
        f.write(raw_df.dtypes.to_string())
        f.write("\n\n")

        f.write("Missing values before cleaning:\n")
        f.write(pd.Series(missing_before).to_string())
        f.write("\n\n")

        f.write("Basic statistics:\n")
        f.write(clean_df.describe().to_string())
        f.write("\n\n")

        f.write("Age mean/std by outcome:\n")
        f.write(age_stats.to_string())
        f.write("\n\n")

        f.write("Top 10 correlations (absolute sorted):\n")
        f.write(corr_pairs.head(10).to_string(index=False))
        f.write("\n\n")

        f.write("Filtered question answers:\n")
        for key, value in filtered_answers.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")

        f.write("Part 2 - Expert System\n")
        f.write(pd.Series(expert_metrics).to_string())
        f.write("\n\n")

        f.write("Part 3 - Machine Learning Models\n")
        f.write(pd.DataFrame(ml_metrics).T.to_string())
        f.write("\n\n")

        f.write("Justification for cleaning:\n")
        f.write(
            "- Zeros in clinical variables are treated as missing because they are often invalid measurements.\n"
        )
        f.write("- Median imputation is robust for skewed medical variables.\n")
        f.write("- IQR capping reduces effect of extreme outliers without deleting records.\n")


def main() -> None:
    ensure_dirs()
    raw_df = load_data()

    clean_df, missing_before = detect_and_handle_missing_values(raw_df)
    outlier_cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    clean_df = cap_outliers_iqr(clean_df, outlier_cols)

    age_stats = summary_statistics_by_outcome(clean_df)
    corr_pairs = pairwise_correlations_sorted(clean_df)
    filtered_answers = answer_filtered_questions(clean_df)

    plot_task5_figures(clean_df)

    x, y = split_xy(clean_df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    expert_pred = expert_system_predict(x_test)
    expert_metrics = evaluate_predictions(y_test, expert_pred)

    ml_metrics = train_ml_models(x_train, y_train, x_test, y_test)
    all_metrics = {"ExpertSystem": expert_metrics, **ml_metrics}
    plot_task6_comparison(all_metrics)

    save_report(
        raw_df=raw_df,
        clean_df=clean_df,
        missing_before=missing_before,
        age_stats=age_stats,
        corr_pairs=corr_pairs,
        filtered_answers=filtered_answers,
        expert_metrics=expert_metrics,
        ml_metrics=ml_metrics,
    )

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("Done. Outputs generated in ./outputs")
    print(f"- Report: {REPORT_PATH}")
    print(f"- Metrics JSON: {METRICS_PATH}")
    print(f"- Figures folder: {FIG_DIR}")


if __name__ == "__main__":
    main()
