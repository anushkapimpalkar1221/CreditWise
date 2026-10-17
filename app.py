
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction")
st.write(
    "Enter the applicant details below to compare "
    "predictions from three Machine Learning models."
)

# --------------------------------------------------
# Load saved models and preprocessing objects
# --------------------------------------------------
@st.cache_resource
def load_objects():
    logistic_model = joblib.load("loan_model_logistic.pkl")
    knn_model = joblib.load("loan_model_knn.pkl")
    nb_model = joblib.load("loan_model_nb.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("onehot_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return (
        logistic_model,
        knn_model,
        nb_model,
        scaler,
        encoder,
        feature_columns
    )


try:
    (
        logistic_model,
        knn_model,
        nb_model,
        scaler,
        encoder,
        feature_columns
    ) = load_objects()

except Exception as e:
    st.error(f"Could not load the saved model files: {e}")
    st.stop()

# --------------------------------------------------
# Applicant input form
# --------------------------------------------------
st.header("Applicant Information")

with st.form("loan_application_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        applicant_income = st.number_input(
            "Applicant Income", min_value=0.0,
            value=50000.0, step=1000.0
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income", min_value=0.0,
            value=0.0, step=1000.0
        )

        age = st.number_input(
            "Age", min_value=18, max_value=100, value=30
        )

        dependents = st.number_input(
            "Dependents", min_value=0, max_value=20, value=0
        )

        existing_loans = st.number_input(
            "Existing Loans", min_value=0, max_value=50, value=0
        )

        savings = st.number_input(
            "Savings", min_value=0.0,
            value=10000.0, step=1000.0
        )

    with col2:
        collateral_value = st.number_input(
            "Collateral Value", min_value=0.0,
            value=50000.0, step=1000.0
        )

        loan_amount = st.number_input(
            "Loan Amount", min_value=0.0,
            value=20000.0, step=1000.0
        )

        loan_term = st.number_input(
            "Loan Term", min_value=1, value=12
        )

        credit_score = st.number_input(
            "Credit Score", min_value=0.0,
            value=650.0, step=1.0
        )

        dti_ratio = st.number_input(
            "DTI Ratio", min_value=0.0,
            value=0.3, step=0.05, format="%.3f"
        )

        education_level = st.selectbox(
            "Education Level",
            ["Graduate", "Not Graduate"]
        )

    with col3:
        employment_status = st.selectbox(
            "Employment Status",
            ["Contract", "Salaried", "Self-employed", "Unemployed"]
        )

        marital_status = st.selectbox(
            "Marital Status", ["Married", "Single"]
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["Business", "Car", "Education", "Home", "Personal"]
        )

        property_area = st.selectbox(
            "Property Area", ["Rural", "Semiurban", "Urban"]
        )

        gender = st.selectbox("Gender", ["Female", "Male"])

        employer_category = st.selectbox(
            "Employer Category",
            ["Business", "Government", "MNC", "Private", "Unemployed"]
        )

    submitted = st.form_submit_button(
        "Predict Loan Approval",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# Preprocess input and predict
# --------------------------------------------------
if submitted:

    # IMPORTANT:
    # This mapping is a temporary assumption, not verified
    # from your original LabelEncoder.
    education_mapping = {
        "Graduate": 0,
        "Not Graduate": 1
    }

    input_data = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Existing_Loans": existing_loans,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": education_mapping[education_level],
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Gender": gender,
        "Employer_Category": employer_category
    }])

    # Feature engineering: match the training features
    input_data["DTI_Ratio_sq"] = dti_ratio ** 2
    input_data["Credit_Score_sq"] = credit_score ** 2

    categorical_columns = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    # Encode categorical features using the saved encoder
    encoded_array = encoder.transform(
        input_data[categorical_columns]
    )

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(),
        index=input_data.index
    )

    # Keep numeric/engineered columns
    numeric_columns = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Existing_Loans",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
        "Education_Level",
        "DTI_Ratio_sq",
        "Credit_Score_sq"
    ]

    numeric_df = input_data[numeric_columns]

    # Combine features
    final_input = pd.concat(
        [numeric_df, encoded_df],
        axis=1
    )

    # Match the model's exact feature order
    final_input = final_input.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale with the saved scaler
    scaled_input = scaler.transform(final_input)

    # Predict with all three models
    predictions = {
        "Logistic Regression": logistic_model.predict(scaled_input)[0],
        "K-Nearest Neighbors": knn_model.predict(scaled_input)[0],
        "Gaussian Naive Bayes": nb_model.predict(scaled_input)[0]
    }

    # Display results
    st.divider()
    st.header("Prediction Results")

    result_columns = st.columns(3)

    for column, (model_name, prediction) in zip(
        result_columns, predictions.items()
    ):
        with column:
            st.subheader(model_name)

            if prediction == 1:
                st.success("Loan Approved")
            else:
                st.error("Loan Not Approved")

    st.caption(
        "These are model predictions, not a guarantee of actual loan approval."
    )
