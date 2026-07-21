# main.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ==============================

# 1. Load Dataset

# ==============================

df = pd.read_csv("dataset2.csv")

# ==============================

# 2. Data Cleaning

# ==============================

df.drop_duplicates(inplace=True)

df["Gender"] = df["Gender"].fillna("Unknown")
df["Married"] = df["Married"].fillna("No")
df["Dependents"] = df["Dependents"].fillna(0)
df["Self_Employed"] = df["Self_Employed"].fillna("No")
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
df["Loan_Amount_Term(in days)"] = df["Loan_Amount_Term(in days)"].fillna(
    df["Loan_Amount_Term(in days)"].mean()
)
df["Credit_History"] = df["Credit_History"].fillna(1)

# ==============================

# 3. Encoding

# ==============================

le_married = LabelEncoder()
le_self_emp = LabelEncoder()
le_property = LabelEncoder()
le_target = LabelEncoder()

df["Married"] = le_married.fit_transform(df["Married"])
df["Self_Employed"] = le_self_emp.fit_transform(df["Self_Employed"])
df["Property_Area"] = le_property.fit_transform(df["Property_Area"])
df["Loan_Status"] = le_target.fit_transform(df["Loan_Status"])

# ==============================

# 4. Feature Selection

# ==============================

X = df[
    [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term(in days)",
        "Credit_History",
        "Married",
        "Self_Employed",
        "Property_Area",
    ]
]

y = df["Loan_Status"]

# ==============================

# 5. Train-Test Split

# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================

# 6. Scaling

# ==============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================

# 7. Models

# ==============================

lr_model = LogisticRegression(max_iter=200)
knn_model = KNeighborsClassifier(n_neighbors=5)
dt_model = DecisionTreeClassifier(random_state=42)

lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# ==============================

# 8. Evaluation

# ==============================

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_model.predict(X_test)))
print("KNN Accuracy:", accuracy_score(y_test, knn_model.predict(X_test)))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_model.predict(X_test)))

# ==============================

# 9. User Input

# ==============================

print("\n=== Loan Prediction System ===")

App_Income = int(input("Enter Your Monthly Income: "))
Co_App_Income = int(input("Enter Co-Applicant Income: "))
Loan_Amount = int(input("Enter Loan Amount: "))
Loan_Term = int(input("Enter Loan Term (in days): "))
Credit_History = int(input("Credit History (1 = Good, 0 = Bad): "))
Married = int(input("Married (1 = Yes, 0 = No): "))
Self_Employed = int(input("Self Employed (1 = Yes, 0 = No): "))
Property_Area = int(input("Property Area (0 = Rural, 1 = Semiurban, 2 = Urban): "))

user_df = pd.DataFrame(
    {
        "ApplicantIncome": [App_Income],
        "CoapplicantIncome": [Co_App_Income],
        "LoanAmount": [Loan_Amount],
        "Loan_Amount_Term(in days)": [Loan_Term],
        "Credit_History": [Credit_History],
        "Married": [Married],
        "Self_Employed": [Self_Employed],
        "Property_Area": [Property_Area],
    }
)

user_scaled = scaler.transform(user_df)

# ==============================

# 10. Predictions

# ==============================

lr_pred = lr_model.predict(user_scaled)[0]
knn_pred = knn_model.predict(user_scaled)[0]
dt_pred = dt_model.predict(user_scaled)[0]


def result(pred):
    return "Loan Approved ✅" if pred == 1 else "Loan Rejected ❌"


print("\n--- Predictions ---")
print("Logistic Regression:", result(lr_pred))
print("KNN:", result(knn_pred))
print("Decision Tree:", result(dt_pred))

if lr_pred == knn_pred == dt_pred:
    print("\nAll models agree on the result.")
