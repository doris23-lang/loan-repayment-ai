import warnings
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

st.set_page_config(page_title="Loan Repayment Predictor", page_icon="💰", layout="wide")

st.title("AI-Based Loan Repayment Prediction System")
st.markdown(
    "This interactive tool predicts whether a borrower is likely to pay back a loan "
    "based on financial, demographic, and credit-related information."
)

# Load dataset
data = pd.read_csv("loan_dataset_20000.csv")

# Fill missing values
for column in data.columns:
    if is_numeric_dtype(data[column]):
        data[column] = data[column].fillna(data[column].median())
    else:
        data[column] = data[column].fillna(data[column].mode()[0])

target_column = "loan_paid_back"

# Keep original values for UI
original_data = data.copy()

# Encode categorical columns
label_encoders = {}
for column in data.columns:
    if not is_numeric_dtype(data[column]):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

X = data.drop(target_column, axis=1)
y = data[target_column]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

nice_labels = {
    "age": "Age",
    "gender": "Gender",
    "marital_status": "Marital Status",
    "education_level": "Education Level",
    "annual_income": "Annual Income",
    "monthly_income": "Monthly Income",
    "employment_status": "Employment Status",
    "debt_to_income_ratio": "Debt-to-Income Ratio",
    "credit_score": "Credit Score",
    "loan_amount": "Loan Amount",
    "loan_purpose": "Loan Purpose",
    "interest_rate": "Interest Rate",
    "loan_term": "Loan Term",
    "installment": "Installment",
    "grade_subgrade": "Grade / Subgrade",
    "num_of_open_accounts": "Number of Open Accounts",
    "total_credit_limit": "Total Credit Limit",
    "current_balance": "Current Balance",
    "delinquency_history": "Delinquency History",
    "public_records": "Public Records",
    "num_of_delinquencies": "Number of Delinquencies"
}

# Field groups
field_groups = {
    "Borrower Profile": [
        "age", "gender", "marital_status", "education_level"
    ],
    "Income & Employment": [
        "annual_income", "monthly_income", "employment_status", "debt_to_income_ratio"
    ],
    "Loan Details": [
        "loan_amount", "loan_purpose", "interest_rate", "loan_term", "installment"
    ],
    "Credit & Risk History": [
        "credit_score", "grade_subgrade", "num_of_open_accounts",
        "total_credit_limit", "current_balance", "delinquency_history",
        "public_records", "num_of_delinquencies"
    ]
}

st.subheader("Enter Borrower Details")

user_input = {}

def render_input(column_name):
    label = nice_labels.get(column_name, column_name)

    if column_name in label_encoders:
        options = sorted(original_data[column_name].astype(str).unique().tolist())
        selected = st.selectbox(label, options, key=column_name)
        user_input[column_name] = label_encoders[column_name].transform([selected])[0]
    else:
        min_val = float(original_data[column_name].min())
        max_val = float(original_data[column_name].max())
        mean_val = float(original_data[column_name].mean())

        if pd.api.types.is_integer_dtype(original_data[column_name]):
            user_input[column_name] = st.number_input(
                label,
                min_value=int(min_val),
                max_value=int(max_val),
                value=int(mean_val),
                step=1,
                key=column_name
            )
        else:
            user_input[column_name] = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                key=column_name
            )

# Two-column layout for sections
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Borrower Profile")
    for field in field_groups["Borrower Profile"]:
        render_input(field)

    st.markdown("### Income & Employment")
    for field in field_groups["Income & Employment"]:
        render_input(field)

with right_col:
    st.markdown("### Loan Details")
    for field in field_groups["Loan Details"]:
        render_input(field)

    st.markdown("### Credit & Risk History")
    for field in field_groups["Credit & Risk History"]:
        render_input(field)

# Validation checks
validation_messages = []
risk_warnings = []

annual_income = float(user_input.get("annual_income", 0))
monthly_income = float(user_input.get("monthly_income", 0))
loan_amount = float(user_input.get("loan_amount", 0))
credit_score = float(user_input.get("credit_score", 0))
interest_rate = float(user_input.get("interest_rate", 0))
debt_ratio = float(user_input.get("debt_to_income_ratio", 0))
age = float(user_input.get("age", 0))
num_delinquencies = float(user_input.get("num_of_delinquencies", 0))
public_records = float(user_input.get("public_records", 0))

# Hard validation
if annual_income > 0 and monthly_income > annual_income:
    validation_messages.append("Monthly income should not be greater than annual income.")

if age < 18:
    validation_messages.append("Borrower age must be at least 18.")

if credit_score < 300 or credit_score > 850:
    validation_messages.append("Credit score should usually be between 300 and 850.")

if debt_ratio < 0 or debt_ratio > 1:
    validation_messages.append("Debt-to-income ratio should usually be between 0 and 1.")

if interest_rate < 0:
    validation_messages.append("Interest rate cannot be negative.")

if loan_amount <= 0:
    validation_messages.append("Loan amount must be greater than 0.")

if annual_income <= 0:
    validation_messages.append("Annual income must be greater than 0.")

if monthly_income <= 0:
    validation_messages.append("Monthly income must be greater than 0.")

if num_delinquencies < 0:
    validation_messages.append("Number of delinquencies cannot be negative.")

if public_records < 0:
    validation_messages.append("Public records cannot be negative.")

# Warnings
if credit_score < 580:
    risk_warnings.append("Very low credit score detected.")

if debt_ratio > 0.5:
    risk_warnings.append("High debt-to-income ratio detected.")

if num_delinquencies >= 5:
    risk_warnings.append("High delinquency count detected.")

if interest_rate > 20:
    risk_warnings.append("High interest rate detected.")

if annual_income > 0 and loan_amount > annual_income:
    risk_warnings.append("Loan amount is higher than annual income.")

st.markdown("---")

if validation_messages:
    st.error("Please fix the following input issues before predicting:")
    for msg in validation_messages:
        st.write(f"- {msg}")

if risk_warnings:
    st.warning("Risk flags detected:")
    for msg in risk_warnings:
        st.write(f"- {msg}")

input_df = pd.DataFrame([user_input])

# Force correct column order
input_df = input_df[X.columns]

if st.button("Predict Loan Repayment", use_container_width=True):
    if validation_messages:
        st.error("Prediction blocked until invalid inputs are corrected.")
    else:
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.success("Predicted outcome: Loan will likely be paid back.")
        else:
            st.error("Predicted outcome: Loan may not be paid back.")