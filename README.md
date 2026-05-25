# Telco-Customer-Churn

# Explainable Customer Churn Prediction Using XAI

## Project Overview

This project predicts customer churn for a telecommunications company and explains the predictions using Explainable Artificial Intelligence methods.

The business goal is not only to identify customers who are likely to leave, but also to understand why they are at risk. This allows the company to design targeted retention strategies.

## Dataset

Dataset: Telco Customer Churn  
Source: Kaggle  
Target variable: `Churn`

The dataset contains customer-level information such as:

- Demographics
- Contract type
- Internet service
- Payment method
- Monthly charges
- Tenure
- Customer churn status

## Machine Learning Models

The project compares:

1. Logistic Regression as a baseline interpretable model
2. Random Forest as a black-box model

The Random Forest model is used for XAI interpretation.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Recall is especially important because in churn prediction, missing a customer who is likely to leave can be costly for the company.

## XAI Methods

The project uses:

1. Permutation Feature Importance
2. SHAP global explanation
3. SHAP local explanation
4. Simple counterfactual / what-if analysis

## Business Value

The XAI analysis helps answer:

- Which factors increase churn risk?
- Which customers should be targeted first?
- What business actions can reduce churn?
- How can the model support customer retention strategy?

## Repository Structure

```text
telco-churn-xai/
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── telco_churn_xai_report.ipynb
│
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
│
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── explain_xai.py
│   └── run_pipeline.py
│
├── README.md
├── requirements.txt
└── .gitignore