"""Task 5: تحليل واستكشاف بيانات السكري."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "report_task5.txt"


def load_and_clean_data():
    """تحميل البيانات وتنظيف القيم المفقودة والقيم المتطرفة."""
    raw_df = pd.read_csv(DATA_URL)
    clean_df = raw_df.copy()
    missing_before = clean_df.isna().sum()
    medical_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    clean_df[medical_cols] = clean_df[medical_cols].replace(0, np.nan)
    clean_df[medical_cols] = clean_df[medical_cols].fillna(clean_df[medical_cols].median())
    for col in ["Pregnancies", *medical_cols, "DiabetesPedigreeFunction", "Age"]:
        q1, q3 = clean_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        clean_df[col] = clean_df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return raw_df, clean_df, missing_before


def get_sorted_correlations(df: pd.DataFrame):
    """الارتباط بين كل متغيرين مرتبا من الأعلى إلى الأقل."""
    corr = df.corr(numeric_only=True)
    pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i, j]) for i in range(len(corr.columns)) for j in range(i + 1, len(corr.columns))]
    return (
        pd.DataFrame(pairs, columns=["Feature_1", "Feature_2", "Correlation"])
        .assign(Abs_Correlation=lambda x: x["Correlation"].abs())
        .sort_values("Abs_Correlation", ascending=False)
        .reset_index(drop=True)
    )


def answer_filter_questions(df: pd.DataFrame):
    """إجابات الأسئلة المفلترة المطلوبة."""
    answers = {}
    pregnant_35 = df[(df["Pregnancies"] > 0) & (df["Age"] > 35)]
    answers["avg_insulin_pregnant_over_35"] = f"{pregnant_35['Insulin'].mean():.2f}" if not pregnant_35.empty else "لا توجد بيانات"
    if "Gender" in df.columns:
        male = df["Gender"].str.lower().isin(["male", "m", "man"])
        female = df["Gender"].str.lower().isin(["female", "f", "woman"])
        male_diabetic = df[male & (df["Outcome"] == 1)]
        female_non_diabetic = df[female & (df["Outcome"] == 0)]
        male_over_40 = df[male & (df["Age"] > 40)]
        answers["min_bmi_male_diabetic"] = f"{male_diabetic['BMI'].min():.2f}" if not male_diabetic.empty else "لا توجد سجلات"
        answers["bp_range_female_non_diabetic"] = f"{female_non_diabetic['BloodPressure'].min():.2f} - {female_non_diabetic['BloodPressure'].max():.2f}" if not female_non_diabetic.empty else "لا توجد سجلات"
        answers["bp_range_male_over_40"] = f"{male_over_40['BloodPressure'].min():.2f} - {male_over_40['BloodPressure'].max():.2f}" if not male_over_40.empty else "لا توجد سجلات"
    else:
        answers["min_bmi_male_diabetic"] = "غير متاح (لا يوجد Gender)"
        answers["bp_range_female_non_diabetic"] = "غير متاح (لا يوجد Gender)"
        answers["bp_range_male_over_40"] = "غير متاح (لا يوجد Gender)"
    return answers


def create_figures(df: pd.DataFrame):
    """إنشاء الرسوم المطلوبة: Boxplot وBarchart وLine."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    df.boxplot(column="Age", by="Outcome", figsize=(7, 5))
    plt.title("Age Distribution by Diabetes Outcome")
    plt.suptitle("")
    plt.xlabel("Outcome")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "boxplot_age_outcome.png", dpi=180)
    plt.close()
    bmi_df = df.copy()
    bmi_df["BMI_Category"] = pd.cut(bmi_df["BMI"], [0, 18.5, 25, 30, 100], labels=["Underweight", "Normal", "Overweight", "Obese"])
    (bmi_df.groupby("BMI_Category", observed=False)["Outcome"].mean() * 100).plot(kind="bar", figsize=(8, 5))
    plt.title("Diabetes Percentage by BMI Category")
    plt.xlabel("BMI Category")
    plt.ylabel("Diabetes %")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "barchart_bmi_vs_diabetes_rate.png", dpi=180)
    plt.close()
    trend_df = df[["BMI", "Glucose"]].sort_values("BMI").reset_index(drop=True)
    trend_df["Glucose_Rolling"] = trend_df["Glucose"].rolling(25, min_periods=5).mean()
    plt.figure(figsize=(8, 5))
    plt.plot(trend_df["BMI"], trend_df["Glucose_Rolling"])
    plt.title("Trend Between BMI and Glucose")
    plt.xlabel("BMI")
    plt.ylabel("Glucose (rolling mean)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "line_bmi_glucose_trend.png", dpi=180)
    plt.close()


def main():
    raw_df, clean_df, missing_before = load_and_clean_data()
    age_stats = clean_df.groupby("Outcome")["Age"].agg(["mean", "std", "count"])
    corr_pairs = get_sorted_correlations(clean_df)
    answers = answer_filter_questions(clean_df)
    create_figures(clean_df)
    bp_bmi_corr = clean_df["BloodPressure"].corr(clean_df["BMI"])
    glucose_insulin_corr = clean_df["Glucose"].corr(clean_df["Insulin"])
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("=== تقرير المهمة 5 ===\n\n")
        f.write(f"Raw shape: {raw_df.shape}\nClean shape: {clean_df.shape}\n\n")
        f.write("أول 5 سجلات:\n" + raw_df.head().to_string(index=False) + "\n\n")
        f.write("أنواع الأعمدة:\n" + raw_df.dtypes.to_string() + "\n\n")
        f.write("القيم المفقودة قبل التنظيف:\n" + missing_before.to_string() + "\n\n")
        f.write("الإحصاءات الأساسية:\n" + clean_df.describe().to_string() + "\n\n")
        f.write("متوسط/انحراف العمر حسب الإصابة:\n" + age_stats.to_string() + "\n\n")
        f.write(f"ارتباط BloodPressure مع BMI: {bp_bmi_corr:.4f}\n")
        f.write(f"ارتباط Glucose مع Insulin: {glucose_insulin_corr:.4f}\n\n")
        f.write("أعلى 10 معاملات ارتباط:\n" + corr_pairs.head(10).to_string(index=False) + "\n\n")
        f.write("إجابات الأسئلة:\n")
        for key, value in answers.items():
            f.write(f"- {key}: {value}\n")
    print("تم إنجاز المهمة 5.")
    print(f"- التقرير: {REPORT_PATH}")
    print(f"- الرسوم: {FIG_DIR}")


if __name__ == "__main__":
    main()
