# المهمتان 5 و6 - التنبؤ بمرض السكري النوع الثاني

هذا المشروع يقدم حلًا عمليًا كاملًا بملفين منفصلين:

- **المهمة 5:** الملف `task5_analysis.py` (التنظيف، الاستكشاف، التحليل الإحصائي، والرسوم).
- **المهمة 6:** الملف `task6_modeling.py` (النظام الخبير، نماذج التعلم الآلي، والتقييم).

## المتطلبات

تثبيت المكتبات المطلوبة:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## التشغيل

تشغيل المهمة 5:

```bash
python task5_analysis.py
```

تشغيل المهمة 6:

```bash
python task6_modeling.py
```

## المخرجات

بعد التشغيل سيتم حفظ النتائج داخل المجلد `outputs/`:

- `report_task5.txt`: تقرير نصي كامل للمهمة 5.
- `report_task6.txt`: تقرير نصي كامل للمهمة 6.
- `metrics_task6.json`: مقاييس `Accuracy` و`Precision` و`Recall` للنظام الخبير ونماذج التعلم الآلي.
- `figures/`:
  - `boxplot_age_outcome.png`
  - `barchart_bmi_vs_diabetes_rate.png`
  - `line_bmi_glucose_trend.png`
  - `task6_models_comparison.png`

## ملاحظة مهمة بخصوص أسئلة الجنس (ذكر/أنثى)

نسخة بيانات Pima الشائعة لا تحتوي عمود `Gender` وهي أصلًا لعينات نسائية.
لذلك أي أسئلة مقارنة بين الذكور والإناث ستظهر كـ "غير متاح" إلا إذا زودت البيانات بعمود جنس فعلي.
