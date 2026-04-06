# AI-Based Loan Repayment Prediction System

This project uses machine learning to predict whether a borrower is likely to repay a loan using financial and demographic information.

## Objective
The goal of this project is to demonstrate a complete beginner-friendly machine learning workflow:
- loading and exploring data
- preprocessing and encoding variables
- training a classification model
- evaluating performance
- making an individual borrower prediction

## Tools Used
- Python
- Pandas
- Scikit-learn
- Logistic Regression

## Dataset
The dataset was obtained from Kaggle and includes borrower-related features such as:
- age
- income
- employment status
- credit score
- loan amount
- debt-to-income ratio
- delinquency history

Target column:
- `loan_paid_back`
  - `1` = loan paid back
  - `0` = loan not paid back

## Method
1. Loaded the CSV dataset
2. Handled missing values
3. Encoded categorical variables into numeric form
4. Split data into training and test sets
5. Trained a Logistic Regression model
6. Evaluated the model using:
   - accuracy
   - classification report
   - confusion matrix
7. Generated a sample prediction for one borrower

## Result
- Achieved approximately **87.35% accuracy**
- The model shows strong performance in identifying borrowers who repay loans
- Lower recall for defaulters highlights a limitation in detecting high-risk borrowers

## Sample Output
```text
Accuracy: 0.8735
Predicted outcome: Loan will likely be paid back
