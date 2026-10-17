# CreditWise — Loan Approval Prediction

CreditWise is a machine learning project that predicts loan approval using applicant information. It compares three classification algorithms and provides predictions through a Streamlit web application.

## Features

- Loan approval prediction using three machine learning models.
- Exploratory Data Analysis (EDA) and feature engineering.
- Data preprocessing, including encoding and feature scaling.
- Model evaluation using accuracy, precision, recall, and confusion matrices.
- Interactive Streamlit application for comparing model predictions.

## Machine Learning Models

| Model | Accuracy |
|---|---:|
| Logistic Regression | 87.50% |
| Gaussian Naive Bayes | 86.50% |
| K-Nearest Neighbors (KNN) | 75.50% |

*Accuracy values are from the model evaluation performed in this project.*

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn
- Joblib

## Project Structure

```text
CreditWise/
├── app.py
├── credit_wise.ipynb
├── loan_approval_data.csv
├── loan_model_logistic.pkl
├── loan_model_knn.pkl
├── loan_model_nb.pkl
├── scaler.pkl
├── onehot_encoder.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md
```

## How to Run Locally

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Start the Streamlit application:

```bash
streamlit run app.py
```

## Disclaimer

CreditWise provides machine learning predictions for educational and demonstration purposes only. Predictions are estimates and do not guarantee actual loan approval.