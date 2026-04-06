# AI-Based Loan Repayment Prediction System

This project applies machine learning to predict whether a borrower is likely to repay a loan using financial and demographic data. It demonstrates a complete beginner-to-intermediate machine learning workflow, from data preprocessing to model evaluation and prediction.

---

## Objective

The objective of this project is to build and understand a full machine learning pipeline by:

- Loading and exploring structured data  
- Handling missing values  
- Encoding categorical variables  
- Training a classification model  
- Evaluating model performance  
- Making predictions for individual borrowers  

---

## Tools & Technologies

- Python  
- Pandas (data manipulation)  
- Scikit-learn (machine learning)  
- Logistic Regression (classification model)  

---

## Dataset

The dataset used in this project was obtained from Kaggle:

https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025

### File Used:
`loan_dataset_20000.csv`

### Features include:
- Age  
- Income  
- Employment status  
- Credit score  
- Loan amount  
- Debt-to-income ratio  
- Delinquency history  

### Target Variable:
`loan_paid_back`

- `1` → Loan repaid  
- `0` → Loan not repaid  

---

## Methodology

The project follows these steps:

1. **Data Loading**
   - Imported dataset using Pandas  

2. **Data Preprocessing**
   - Handled missing values using median (numerical) and mode (categorical)  
   - Converted categorical variables into numeric form using Label Encoding  

3. **Feature Selection**
   - Separated input features (`X`) and target variable (`y`)  

4. **Train-Test Split**
   - Split dataset into training (80%) and testing (20%) sets  

5. **Model Training**
   - Trained a Logistic Regression model for binary classification  

6. **Model Evaluation**
   - Accuracy score  
   - Classification report  
   - Confusion matrix  

7. **Prediction**
   - Generated prediction for an individual borrower  

---

## Results

- Model Accuracy: **87.35%**

### Key Insights:
- The model performs well in predicting borrowers who are likely to repay loans  
- Lower recall for non-repayment cases indicates reduced sensitivity to high-risk borrowers  
- This reflects a common real-world challenge in credit risk modeling (class imbalance)  

---

## Sample Output

```text
Accuracy: 0.8735
Predicted outcome: Loan will likely be paid back
```


## How to Run

### 1. Download the Dataset

Download the dataset from Kaggle:  
https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025

Extract the file and locate:

`loan_dataset_20000.csv`

---

### 2. Place the Dataset

Move the file `loan_dataset_20000.csv` into the same folder as:

`loan_prediction.py`

---

### 3. Install Required Libraries

Open Command Prompt (or Terminal) and run:

```bash
pip install -r requirements.txt

```

### 4. Run the Project
In the same folder, run:

```bash
python loan_prediction.py

```

### 5. View Results

The program will display:

- Model accuracy
- Classification report
- Confusion matrix
- A prediction for a sample borrower
