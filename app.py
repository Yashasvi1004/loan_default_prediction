import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💳", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load('xgb_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, feature_columns

model, feature_columns = load_artifacts()

st.title("💳 Loan Default Risk Predictor")
st.caption("XGBoost model trained on 255K+ loan records (demo project — not for real lending decisions)")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 75, 35)
    income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=20000, step=500)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    months_employed = st.slider("Months Employed", 0, 480, 60)
    num_credit_lines = st.slider("Number of Credit Lines", 1, 4, 2)

with col2:
    interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 10.0, step=0.1)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35, step=0.01)
    education = st.selectbox("Education", ["Bachelor's", "High School", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])

st.markdown("---")
col3, col4, col5, col6 = st.columns(4)
with col3:
    has_mortgage = st.checkbox("Has Mortgage")
with col4:
    has_dependents = st.checkbox("Has Dependents")
with col5:
    has_cosigner = st.checkbox("Has Co-Signer")
with col6:
    loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])

st.markdown("---")

if st.button("Predict Default Risk", type="primary", use_container_width=True):
    # Build a single-row dataframe matching training feature columns
    row = {col: 0 for col in feature_columns}
    row['Age'] = age
    row['Income'] = income
    row['LoanAmount'] = loan_amount
    row['CreditScore'] = credit_score
    row['MonthsEmployed'] = months_employed
    row['NumCreditLines'] = num_credit_lines
    row['InterestRate'] = interest_rate
    row['LoanTerm'] = loan_term
    row['DTIRatio'] = dti_ratio
    row['HasMortgage'] = int(has_mortgage)
    row['HasDependents'] = int(has_dependents)
    row['HasCoSigner'] = int(has_cosigner)

    edu_col = f"Education_{education}"
    if edu_col in row:
        row[edu_col] = 1

    emp_col = f"EmploymentType_{employment_type}"
    if emp_col in row:
        row[emp_col] = 1

    mar_col = f"MaritalStatus_{marital_status}"
    if mar_col in row:
        row[mar_col] = 1

    purpose_col = f"LoanPurpose_{loan_purpose}"
    if purpose_col in row:
        row[purpose_col] = 1

    input_df = pd.DataFrame([row])[feature_columns]

    prob = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    st.markdown("### Result")
    risk_pct = prob * 100

    if pred == 1:
        st.error(f"⚠️ Predicted: **Likely to Default**  \nDefault probability: **{risk_pct:.1f}%**")
    else:
        st.success(f"✅ Predicted: **Likely to Repay**  \nDefault probability: **{risk_pct:.1f}%**")

    st.progress(min(int(risk_pct), 100))

    st.caption(
        "This model has a recall of ~67% and precision of ~23% on the minority (default) class — "
        "it's a decision-support demo, not a production lending tool. "
        "Built to showcase the full ML pipeline: imbalance handling, leakage-aware preprocessing, and model evaluation."
    )

st.markdown("---")
st.caption("Model: XGBoost (untuned) | scale_pos_weight for class imbalance | ROC-AUC ≈ 0.76 on held-out test set")
