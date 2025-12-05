import numpy as np
import pandas as pd

# لتثبيت العشوائية
np.random.seed(42)

n = 50  # عدد الطلاب

# أعمدة أساسية
student_id = list(range(1, n+1))
sex = np.random.choice(['M', 'F'], size=n)
age = np.random.randint(15, 19, size=n)  # أعمار 15-18
studytime = np.random.choice([1, 2, 3, 4], size=n, p=[0.45, 0.35, 0.15, 0.05])
failures = np.random.choice([0, 0, 0, 1, 2, 3], size=n)  # أغلب الطلاب ما رسبوا
absences = np.random.poisson(3, size=n)  # توزيع طبيعي للغيابات

# الدرجات: G1 و G2
g1 = np.clip(np.round(np.random.normal(11, 3, size=n)), 0, 20).astype(int)
g2 = np.clip(np.round(g1 + np.random.normal(0.5, 2, size=n)), 0, 20).astype(int)

# الدرجة النهائية G3 محسوبة من G1, G2 + عوامل أخرى + noise
g3_cont = (
    0.35 * g1
    + 0.45 * g2
    + 0.8 * studytime
    - 0.25 * failures
    - 0.15 * absences
    + np.random.normal(0, 1.2, size=n)
)
g3 = np.clip(np.round(g3_cont), 0, 20).astype(int)

# المتغير الهدف: نجح / رسب
passed = (g3 >= 10).astype(int)

# بناء DataFrame
df = pd.DataFrame({
    'student_id': student_id,
    'sex': sex,
    'age': age,
    'studytime': studytime,
    'failures': failures,
    'absences': absences,
    'G1': g1,
    'G2': g2,
    'G3': g3,
    'passed': passed
})

# حفظ الملف
df.to_csv("student_exam.csv", index=False)

print(df.head())

