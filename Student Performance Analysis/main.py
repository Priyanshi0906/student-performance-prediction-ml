
#   STUDENT PERFORMANCE PREDICTION SYSTEM
#   Linear Regression (Score) + Logistic Regression (Pass/Fail)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

#  1. LOAD DATASET and CLEAN DATASET

df = pd.read_csv('Student_Performance.csv')
print("Shape:", df.shape)
print(df.head())

df = df.drop_duplicates()
print("Missing values before cleaning:\n", df.isnull().sum())

df = df[(df['study_hours'] >= 0) & (df['study_hours'] <= 12)]
df = df[(df['attendance_percentage'] >= 0) & (df['attendance_percentage'] <= 100)]

print("Shape after cleaning:", df.shape)


# Create pass/fail column (Pass = 1, Fail = 0)
df['pass_fail'] = df['final_grade'].apply(lambda x: 1 if x in ['a', 'b', 'c'] else 0)

# Encode categorical columns
le = LabelEncoder()
cat_cols = ['gender', 'school_type', 'parent_education',
            'internet_access', 'travel_time',
            'extra_activities', 'study_method']

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Features and targets
X = df[['age', 'gender', 'school_type', 'parent_education',
        'study_hours', 'attendance_percentage',
        'internet_access', 'travel_time',
        'extra_activities', 'study_method']]

y_cls = df['pass_fail']        # for classification
y_reg = df['overall_score']   # for regression

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data preprocessing done!")

#  3. EXPLORATORY DATA ANALYSIS (EDA)

# 3.1 Grade Distribution
plt.figure(figsize=(7, 4))
grade_counts = df['final_grade'].str.upper().value_counts().sort_index()
plt.bar(grade_counts.index, grade_counts.values, color='steelblue', edgecolor='black')
plt.title('Grade Distribution')
plt.xlabel('Grade')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.savefig("Grade_Distribution.png")
plt.show()

# 3.2 Pass / Fail Count
plt.figure(figsize=(5, 4))
df['pass_fail'].value_counts().plot(kind='bar', color=['salmon', 'mediumseagreen'],
                                    edgecolor='black')
plt.title('Pass / Fail Count')
plt.xlabel('0 = Fail  |  1 = Pass')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("pass_fail_count.png")
plt.show()

# 3.3 Study Hours vs Overall Score
plt.figure(figsize=(7, 4))
plt.scatter(df['study_hours'], df['overall_score'],
            c=df['pass_fail'], cmap='RdYlGn', alpha=0.4, s=10)
m, b = np.polyfit(df['study_hours'], df['overall_score'], 1)
x_line = np.linspace(df['study_hours'].min(), df['study_hours'].max(), 100)
plt.plot(x_line, m * x_line + b, color='blue', linewidth=2, linestyle='--', label='Trend Line')
plt.title('Study Hours vs Overall Score')
plt.xlabel('Study Hours')
plt.ylabel('Overall Score')
plt.legend()
plt.tight_layout()
plt.savefig('Study_Hours_vs_Overall_Score.png')
plt.show()

# 3.4 Attendance % vs Overall Score
plt.figure(figsize=(7, 4))
plt.scatter(df['attendance_percentage'], df['overall_score'],
            c=df['pass_fail'], cmap='coolwarm', alpha=0.3, s=10)
m2, b2 = np.polyfit(df['attendance_percentage'], df['overall_score'], 1)
x_line2 = np.linspace(df['attendance_percentage'].min(), df['attendance_percentage'].max(), 100)
plt.plot(x_line2, m2 * x_line2 + b2, color='blue', linewidth=2, linestyle='--', label='Trend Line')
plt.title('Attendance % vs Overall Score')
plt.xlabel('Attendance Percentage')
plt.ylabel('Overall Score')
plt.legend()
plt.tight_layout()
plt.savefig('Attendance_%_vs_Overall_Score.png')
plt.show()

# 3.5 Average Score by Study Method
plt.figure(figsize=(8, 4))
sm_avg = df.groupby('study_method')['overall_score'].mean().sort_values()
plt.barh(sm_avg.index, sm_avg.values, color='cornflowerblue', edgecolor='black')
plt.title('Average Score by Study Method')
plt.xlabel('Average Overall Score')
plt.tight_layout()
plt.savefig(' Average_Score_by_Study_Method.png')
plt.show()

# 3.6 Score Distribution by Gender
plt.figure(figsize=(7, 4))

gender_map = {0: 'Male', 1: 'Female', 2: 'Other'}

for gender in df['gender'].unique():
    subset = df[df['gender'] == gender]['overall_score']
    plt.hist(subset, bins=30, alpha=0.5,
             edgecolor='black',
             label=gender_map.get(gender, str(gender)))

plt.title('Score Distribution by Gender')
plt.xlabel('Overall Score')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('Score_Distribution_by_Gender.png')
plt.show()


#  4. FEATURE SCALING 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  5. TRAIN/TEST SPLIT 

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_cls, test_size=0.2, stratify=y_cls, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, stratify=y_cls, random_state=42)


# 6. TRAIN MODELS 

lr = LinearRegression()
lr.fit(X_train_r, y_train_r)

log = LogisticRegression(max_iter=1000)
log.fit(X_train_c, y_train_c)

# 7. Predictions
y_pred_r = lr.predict(X_test_r)
y_pred_c = log.predict(X_test_c)

#  8. EVALUATE

print("Linear Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test_r, y_pred_r)))
print("R2 Score:", r2_score(y_test_r, y_pred_r))

print("\n Logistic Regression:")
print("Accuracy:", accuracy_score(y_test_c, y_pred_c))


#  8.1 Actual vs Predicted Plot (Linear Regression) 
plt.figure(figsize=(6, 4))
plt.scatter(y_test_r, y_pred_r, alpha=0.5, color='steelblue', edgecolors='black', s=20)
min_val = min(y_test_r.min(), y_pred_r.min())
max_val = max(y_test_r.max(), y_pred_r.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted (Linear Regression)")
plt.legend()
plt.tight_layout()
plt.savefig('Actual_vs_Predicted_Plot.png')
plt.show()

# 8.2 Residual Plot 
plt.figure(figsize=(6, 4))
residuals = y_test_r - y_pred_r
plt.hist(residuals, bins=50, color='orange', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residual Distribution (Linear Regression)')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Residual_Plot.png')
plt.show()


#  8.3 Confusion Matrix (Logistic Regression) 
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test_c, y_pred_c)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail', 'Pass'],
            yticklabels=['Fail', 'Pass'])
plt.title('Confusion Matrix (Logistic Regression)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('Confusion_Matrix.png')
plt.show()


#  9. PREDICT A NEW STUDENT 
new_student = pd.DataFrame([{
  'age': 17,
    'gender': 1,
    'school_type': 0,
    'parent_education': 2,
    'study_hours': 6.0,
    'attendance_percentage': 85.0,
    'internet_access': 1,
    'travel_time': 0,
    'extra_activities': 1,
    'study_method': 4
}])
new_scaled = scaler.transform(new_student)
pred_score = lr.predict(new_scaled)[0]
pred_pass  = log.predict(new_scaled)[0]
pred_prob  = log.predict_proba(new_scaled)[0][1]

print(f"\n── PREDICTION FOR NEW STUDENT ──")
print(f"  Predicted Score  : {pred_score:.1f} / 100")
print(f"  Pass/Fail        : {'PASS' if pred_pass==1 else 'FAIL'}")
print(f"  Pass Probability : {pred_prob*100:.1f}%")


