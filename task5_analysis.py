"""المهمة 5: تحليل بسيط جدا لبيانات السكري."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "report_task5.txt"

OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# 1) تحميل البيانات
raw_df = pd.read_csv(DATA_URL)
df = raw_df.copy()
missing_before = df.isna().sum()

# 2) تنظيف بسيط
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# 3) حسابات أساسية
age_stats = df.groupby("Outcome")["Age"].agg(["mean", "std", "count"])
bp_bmi_corr = df["BloodPressure"].corr(df["BMI"])
glucose_insulin_corr = df["Glucose"].corr(df["Insulin"])

corr = df.corr(numeric_only=True)
pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
corr_pairs = pd.DataFrame(pairs, columns=["Feature_1", "Feature_2", "Correlation"])
corr_pairs["Abs_Correlation"] = corr_pairs["Correlation"].abs()
corr_pairs = corr_pairs.sort_values("Abs_Correlation", ascending=False)

# 4) إجابات الأسئلة
answers = {}
pregnant_35 = df[(df["Pregnancies"] > 0) & (df["Age"] > 35)]
if len(pregnant_35) > 0:
    answers["avg_insulin_pregnant_over_35"] = f"{pregnant_35['Insulin'].mean():.2f}"
else:
    answers["avg_insulin_pregnant_over_35"] = "لا توجد بيانات"

answers["min_bmi_male_diabetic"] = "غير متاح (لا يوجد Gender)"
answers["bp_range_female_non_diabetic"] = "غير متاح (لا يوجد Gender)"
answers["bp_range_male_over_40"] = "غير متاح (لا يوجد Gender)"

# 5) الرسوم
df.boxplot(column="Age", by="Outcome", figsize=(7, 5))
plt.title("Age Distribution by Diabetes Outcome")
plt.suptitle("")
plt.xlabel("Outcome")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig(FIG_DIR / "boxplot_age_outcome.png", dpi=180)
plt.close()

bmi_df = df.copy()
bmi_df["BMI_Category"] = pd.cut(
    bmi_df["BMI"], [0, 18.5, 25, 30, 100], labels=["Underweight", "Normal", "Overweight", "Obese"]
)
(bmi_df.groupby("BMI_Category", observed=False)["Outcome"].mean() * 100).plot(kind="bar", figsize=(8, 5))
plt.title("Diabetes Percentage by BMI Category")
plt.xlabel("BMI Category")
plt.ylabel("Diabetes %")
plt.tight_layout()
plt.savefig(FIG_DIR / "barchart_bmi_vs_diabetes_rate.png", dpi=180)
plt.close()

trend_df = df[["BMI", "Glucose"]].sort_values("BMI").reset_index(drop=True)
trend_df["Glucose_Rolling"] = trend_df["Glucose"].rolling(20, min_periods=5).mean()
plt.figure(figsize=(8, 5))
plt.plot(trend_df["BMI"], trend_df["Glucose_Rolling"])
plt.title("Trend Between BMI and Glucose")
plt.xlabel("BMI")
plt.ylabel("Glucose (rolling mean)")
plt.tight_layout()
plt.savefig(FIG_DIR / "line_bmi_glucose_trend.png", dpi=180)
plt.close()

# 6) التقرير
with REPORT_PATH.open("w", encoding="utf-8") as f:
    f.write("=== تقرير المهمة 5 (نسخة ابسط) ===\n\n")
    f.write("الفكرة:\n")
    f.write("تحليل مباشر للبيانات وتنظيف بسيط مع 3 رسومات.\n\n")
    f.write(f"حجم البيانات قبل التنظيف: {raw_df.shape}\n")
    f.write(f"حجم البيانات بعد التنظيف: {df.shape}\n\n")
    f.write("أول 5 سجلات:\n" + raw_df.head().to_string(index=False) + "\n\n")
    f.write("أنواع الأعمدة:\n" + raw_df.dtypes.to_string() + "\n\n")
    f.write("القيم المفقودة قبل التنظيف:\n" + missing_before.to_string() + "\n\n")
    f.write("إحصاءات أساسية:\n" + df.describe().to_string() + "\n\n")
    f.write("متوسط/انحراف العمر حسب الإصابة:\n" + age_stats.to_string() + "\n\n")
    f.write(f"ارتباط BloodPressure مع BMI: {bp_bmi_corr:.4f}\n")
    f.write(f"ارتباط Glucose مع Insulin: {glucose_insulin_corr:.4f}\n\n")
    f.write("أعلى 10 معاملات ارتباط:\n" + corr_pairs.head(10).to_string(index=False) + "\n\n")
    f.write("إجابات الأسئلة المطلوبة:\n")
    for key, value in answers.items():
        f.write(f"- {key}: {value}\n")

print("تم إنهاء المهمة 5 بشكل ابسط.")
print(f"- التقرير: {REPORT_PATH}")
print(f"- الرسوم: {FIG_DIR}")
