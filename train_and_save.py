import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# ---- Load ----
df = pd.read_csv('Loan_default.csv')
df.drop(columns='LoanID', inplace=True)

# ---- Encode (exact same steps as notebook) ----
binary_map = {'No': 0, 'Yes': 1}
binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
for col in binary_cols:
    df[col] = df[col].map(binary_map)

ohe_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)

# ---- Split ----
X = df.drop(columns='Default')
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Train untuned XGBoost (with scale_pos_weight, no SMOTE) ----
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)

# ---- Evaluate (sanity check vs your notebook numbers) ----
pred = xgb.predict(X_test)
prob = xgb.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred))
print("Accuracy :", accuracy_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1-score :", f1_score(y_test, pred))
print("ROC-AUC  :", roc_auc_score(y_test, prob))

# ---- Save model + feature column order ----
joblib.dump(xgb, 'xgb_model.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')
print("\nSaved xgb_model.pkl and feature_columns.pkl")
print("Feature columns:", list(X.columns))
