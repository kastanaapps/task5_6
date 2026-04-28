"""المهمة 6: نظام خبير + 3 نماذج تعلم آلي للتنبؤ بالسكري."""

import json
from pathlib import Path

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
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
REPORT = OUT_DIR / "report_task6.txt"
METRICS = OUT_DIR / "metrics_task6.json"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """تنظيف مختصر مطابق للمهمة 5."""
    df = df.copy()
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols] = df[cols].replace(0, np.nan)
    df[cols] = df[cols].fillna(df[cols].median())
    for c in ["Pregnancies", *cols, "DiabetesPedigreeFunction", "Age"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df


def eval_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def expert_predict(x: pd.DataFrame):
    """قاعدة خبرة: مصاب إذا تحقق شرطان أو أكثر."""
    score = (
        (x["Glucose"] >= 125).astype(int)
        + (x["BMI"] >= 30).astype(int)
        + (x["Age"] >= 45).astype(int)
        + (x["DiabetesPedigreeFunction"] >= 0.6).astype(int)
        + (x["Insulin"] >= 150).astype(int)
    )
    return (score >= 2).astype(int).values


def main():
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    df = clean_data(pd.read_csv(DATA_URL))
    x, y = df.drop(columns=["Outcome"]), df["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # Expert System
    expert_m = eval_metrics(y_test, expert_predict(x_test))

    # ML models
    scaler = StandardScaler()
    x_train_s, x_test_s = scaler.fit_transform(x_train), scaler.transform(x_test)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=9),
    }
    ml = {}
    for name, model in models.items():
        if name in {"LogisticRegression", "KNN"}:
            model.fit(x_train_s, y_train)
            pred = model.predict(x_test_s)
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
        ml[name] = eval_metrics(y_test, pred)

    all_m = {"ExpertSystem": expert_m, **ml}

    # رسم المقارنة
    pd.DataFrame(all_m).T.plot(kind="bar", figsize=(9, 5))
    plt.title("Model Comparison (Accuracy / Precision / Recall)")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task6_models_comparison.png", dpi=180)
    plt.close()

    # حفظ المخرجات
    with METRICS.open("w", encoding="utf-8") as f:
        json.dump(all_m, f, indent=2)
    with REPORT.open("w", encoding="utf-8") as f:
        f.write("=== تقرير المهمة 6 ===\n\n")
        f.write(f"Shape after cleaning: {df.shape}\n\n")
        f.write("Expert metrics:\n" + pd.Series(expert_m).to_string() + "\n\n")
        f.write("ML metrics:\n" + pd.DataFrame(ml).T.to_string() + "\n")

    print("تم إنجاز المهمة 6.")
    print(f"- التقرير: {REPORT}")
    print(f"- المقاييس: {METRICS}")
    print(f"- الرسم: {FIG_DIR / 'task6_models_comparison.png'}")


if __name__ == "__main__":
    main()
