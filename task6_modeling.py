"""المهمة 6: مقارنة بسيطة جدا بين النظام الخبير ونماذج ML."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
REPORT = OUT_DIR / "report_task6.txt"
METRICS = OUT_DIR / "metrics_task6.json"


def eval_metrics(y_true, y_pred):
    """حساب المقاييس المطلوبة."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def expert_predict(x):
    """نظام خبير بسيط: إذا تحقق شرطان أو أكثر."""
    score = (
        (x["Glucose"] >= 125).astype(int)
        + (x["BMI"] >= 30).astype(int)
        + (x["Age"] >= 45).astype(int)
        + (x["Insulin"] >= 150).astype(int)
    )
    return (score >= 2).astype(int)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_URL).copy()
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    df = df.drop_duplicates().reset_index(drop=True)

    x = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    all_metrics = {}

    # 1) النظام الخبير
    pred_expert = expert_predict(x_test)
    all_metrics["ExpertSystem"] = eval_metrics(y_test, pred_expert)

    # 2) Logistic Regression
    model_lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model_lr.fit(x_train, y_train)
    pred_lr = model_lr.predict(x_test)
    all_metrics["LogisticRegression"] = eval_metrics(y_test, pred_lr)

    # 3) Decision Tree
    model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    model_dt.fit(x_train, y_train)
    pred_dt = model_dt.predict(x_test)
    all_metrics["DecisionTree"] = eval_metrics(y_test, pred_dt)

    # 4) Random Forest
    model_rf = RandomForestClassifier(n_estimators=150, random_state=42)
    model_rf.fit(x_train, y_train)
    pred_rf = model_rf.predict(x_test)
    all_metrics["RandomForest"] = eval_metrics(y_test, pred_rf)

    # 5) KNN
    model_knn = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=9)),
        ]
    )
    model_knn.fit(x_train, y_train)
    pred_knn = model_knn.predict(x_test)
    all_metrics["KNN"] = eval_metrics(y_test, pred_knn)

    # رسم مقارنة المقاييس
    pd.DataFrame(all_metrics).T.plot(kind="bar", figsize=(9, 5))
    plt.title("Simple Models Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task6_models_comparison.png", dpi=180)
    plt.close()

    # حفظ النتائج
    with METRICS.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    with REPORT.open("w", encoding="utf-8") as f:
        f.write("=== تقرير المهمة 6 (نسخة ابسط) ===\n\n")
        f.write("الفكرة:\n")
        f.write("استخدمت نظام خبير بسيط + 4 نماذج تعلم آلي.\n")
        f.write("ثم قارنت بينهم في accuracy و precision و recall و f1.\n\n")
        f.write(f"حجم البيانات بعد التنظيف: {df.shape}\n\n")
        f.write("النتائج:\n")
        f.write(pd.DataFrame(all_metrics).T.to_string())
        f.write("\n\n")
        f.write("ملاحظة بسيطة:\n")
        f.write("الهدف هنا شرح واضح ومباشر بدون تعقيد كبير في الضبط.\n")

    print("تم إنهاء المهمة 6 بشكل مبسط.")
    print(f"التقرير: {REPORT}")
    print(f"المقاييس: {METRICS}")
    print(f"الرسم: {FIG_DIR / 'task6_models_comparison.png'}")


if __name__ == "__main__":
    main()
