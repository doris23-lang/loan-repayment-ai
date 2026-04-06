import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 1. Load dataset
data = pd.read_csv("loan_dataset_20000.csv")

# 2. Preview data
print("First 5 rows:")
print(data.head())

print("\nColumn names:")
print(data.columns)

print("\nData types:")
print(data.dtypes)

# 3. Fill missing values safely
for column in data.columns:
    if is_numeric_dtype(data[column]):
        data[column] = data[column].fillna(data[column].median())
    else:
        data[column] = data[column].fillna(data[column].mode()[0])

# 4. Convert text columns to numbers
le = LabelEncoder()
for column in data.columns:
    if not is_numeric_dtype(data[column]):
        data[column] = le.fit_transform(data[column])

# 5. Set target column
target_column = "loan_paid_back"

# 6. Separate inputs and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# 9. Predict
predictions = model.predict(X_test)

# 10. Results
print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

sample_customer = X_test.iloc[[0]]
sample_prediction = model.predict(sample_customer)

print("\nSample Prediction for One Customer:")
if sample_prediction[0] == 1:
    print("Predicted outcome: Loan will likely be paid back")
else:
    print("Predicted outcome: Loan may not be paid back")
