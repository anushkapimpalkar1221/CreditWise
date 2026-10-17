
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CreditWise | Loan Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- PREMIUM DARK UI ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, #142c4b 0%, transparent 38%),
        radial-gradient(ellipse at 90% 12%, #102d35 0%, transparent 30%),
        #080f1e;
    color: #edf4ff;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, p, label {
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3 { color: #edf4ff !important; }
p, label { color: #c2cee1; }

.brand {
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 2px;
    color: #38dfc0;
}

.hero { padding: 1.8rem 0 1.3rem; }

.hero h1 {
    font-size: clamp(2.1rem, 4vw, 3.4rem);
    font-weight: 800;
    letter-spacing: -1.8px;
    margin: 0.6rem 0;
}

.hero p {
    color: #8fa4c3;
    font-size: 1rem;
    line-height: 1.8;
}

.eyebrow {
    color: #38dfc0;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
}

.panel {
    background: linear-gradient(145deg, #14233a, #0d1729);
    border: 1px solid #243550;
    border-radius: 18px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
}

.panel-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #edf4ff;
    margin-bottom: 0.3rem;
}

.panel-subtitle {
    font-size: 0.85rem;
    color: #8fa4c3;
}

.metric-card {
    background: linear-gradient(145deg, #142640, #0e192c);
    border: 1px solid #2b4261;
    border-radius: 16px;
    padding: 1.2rem;
    min-height: 125px;
    box-shadow: 0 0 22px rgba(57, 135, 255, 0.07);
}

.metric-label {
    color: #9bb0cc;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
}

.metric-value {
    color: #f2f7ff;
    font-size: 1.8rem;
    font-weight: 800;
}

.metric-note {
    color: #38dfc0;
    font-size: 0.76rem;
    margin-top: 0.4rem;
}

div[data-testid="stForm"] {
    background: rgba(15, 27, 46, 0.82);
    border: 1px solid #243550;
    border-radius: 18px;
    padding: 1.4rem;
}

div[data-testid="stForm"] label {
    color: #c8d7ed !important;
    font-size: 0.85rem !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #0a1425 !important;
    border: 1px solid #2a3d59 !important;
    border-radius: 9px !important;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] * {
    color: #edf4ff !important;
}

.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #327dff, #16bba9);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 3rem;
}

div[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 0 22px rgba(56, 223, 192, 0.22);
}

.result-card {
    background: linear-gradient(150deg, #142640, #0d1728);
    border: 1px solid #2a405e;
    border-radius: 16px;
    padding: 1.3rem;
    min-height: 160px;
    box-shadow: 0 0 25px rgba(50, 125, 255, 0.07);
}

.result-model {
    color: #a7bad5;
    font-size: 0.82rem;
    font-weight: 600;
}

.result-status {
    font-size: 1.2rem;
    font-weight: 800;
    margin: 0.9rem 0 0.5rem;
}

.approved { color: #38dfc0; }
.rejected { color: #ff879b; }

.footer {
    text-align: center;
    color: #7186a5;
    font-size: 0.78rem;
    padding-top: 2rem;
    border-top: 1px solid #22324b;
    margin-top: 2rem;
}

hr { border-color: #233650; }

@media (max-width: 700px) {
    .block-container { padding: 1rem 1rem 2rem; }
    .hero h1 { letter-spacing: -1px; }
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown(
    '<div class="brand">💠 CREDITWISE '
    '<span style="color:#7186a5;font-weight:500;letter-spacing:1px;">'
    '/ LOAN INTELLIGENCE</span></div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
    <div class="eyebrow">SMARTER DECISIONS START HERE</div>
    <h1>Loan Approval<br>Prediction.</h1>
    <p>
        Analyze applicant information and compare predictions from three
        machine learning models in one place.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- MODEL PERFORMANCE DATA ----------------
# Previously measured test-set results.
# Update these values if your model evaluation results change.

accuracy_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Gaussian Naive Bayes",
        "K-Nearest Neighbors"
    ],
    "Accuracy (%)": [87.50, 86.50, 75.50]
})

metrics_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Gaussian Naive Bayes",
        "K-Nearest Neighbors"
    ],
    "Precision (%)": [87.56, 86.44, 74.51],
    "Recall (%)": [87.50, 86.50, 75.50]
})


# ---------------- TOP METRIC CARDS ----------------
metric_cols = st.columns(3)

model_metrics = [
    ("LOGISTIC REGRESSION", "87.50%", "Highest measured accuracy"),
    ("GAUSSIAN NAIVE BAYES", "86.50%", "Measured test accuracy"),
    ("K-NEAREST NEIGHBORS", "75.50%", "Measured test accuracy")
]

for col, (name, score, note) in zip(metric_cols, model_metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{name}</div>
            <div class="metric-value">{score}</div>
            <div class="metric-note">✦ {note}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")


# ---------------- MODEL ANALYSIS GRAPHS ----------------
st.markdown("""
<div class="panel">
    <div class="panel-title">◈ Model Performance Analytics</div>
    <div class="panel-subtitle">
        Compare the evaluation results measured during model testing.
        These charts show overall test performance, not the current
        applicant's prediction.
    </div>
</div>
""", unsafe_allow_html=True)


# Accuracy comparison chart
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    accuracy_fig = px.bar(
        accuracy_df,
        x="Model",
        y="Accuracy (%)",
        color="Model",
        text="Accuracy (%)",
        color_discrete_sequence=["#5295ff", "#38dfc0", "#7c8cff"],
        template="plotly_dark",
        title="Accuracy Comparison"
    )

    accuracy_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        cliponaxis=False
    )

    accuracy_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#edf4ff",
        showlegend=False,
        yaxis_range=[0, 100],
        margin=dict(l=10, r=10, t=55, b=10)
    )

    st.plotly_chart(accuracy_fig, use_container_width=True)


# Precision vs Recall chart
with chart_col2:
    metrics_long = metrics_df.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score (%)"
    )

    metrics_fig = px.bar(
        metrics_long,
        x="Model",
        y="Score (%)",
        color="Metric",
        barmode="group",
        text="Score (%)",
        color_discrete_map={
            "Precision (%)": "#5295ff",
            "Recall (%)": "#38dfc0"
        },
        template="plotly_dark",
        title="Precision vs. Recall"
    )

    metrics_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        cliponaxis=False
    )

    metrics_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#edf4ff",
        yaxis_range=[0, 105],
        margin=dict(l=10, r=10, t=55, b=10)
    )

    st.plotly_chart(metrics_fig, use_container_width=True)


# ---------------- LOAD SAVED MODELS ----------------
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


# ---------------- APPLICATION FORM ----------------
st.markdown("""
<div class="panel">
    <div class="panel-title">◈ Applicant Profile</div>
    <div class="panel-subtitle">
        Enter the details below to generate predictions from all three models.
    </div>
</div>
""", unsafe_allow_html=True)


with st.form("loan_application_form"):

    st.markdown("#### 01 / Financial Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=0.0,
            step=1000.0
        )

        savings = st.number_input(
            "Savings",
            min_value=0.0,
            value=10000.0,
            step=1000.0
        )

    with col2:
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )

        loan_term = st.number_input(
            "Loan Term",
            min_value=1,
            value=12
        )

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=50,
            value=0
        )

    with col3:
        collateral_value = st.number_input(
            "Collateral Value",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=0.0,
            value=650.0,
            step=1.0
        )

        dti_ratio = st.number_input(
            "DTI Ratio",
            min_value=0.0,
            value=0.3,
            step=0.05,
            format="%.3f"
        )

    st.markdown("#### 02 / Personal Information")

    col4, col5, col6 = st.columns(3)

    with col4:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=20,
            value=0
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    with col5:
        education_level = st.selectbox(
            "Education Level",
            ["Graduate", "Not Graduate"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Single"]
        )

        employment_status = st.selectbox(
            "Employment Status",
            ["Contract", "Salaried", "Self-employed", "Unemployed"]
        )

    with col6:
        employer_category = st.selectbox(
            "Employer Category",
            ["Business", "Government", "MNC", "Private", "Unemployed"]
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["Business", "Car", "Education", "Home", "Personal"]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Rural", "Semiurban", "Urban"]
        )

    st.write("")

    submitted = st.form_submit_button(
        "✦  Analyze Loan Application",
        type="primary",
        use_container_width=True
    )


# ---------------- PREDICTION LOGIC ----------------
if submitted:

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

    # Engineered features
    input_data["DTI_Ratio_sq"] = dti_ratio ** 2
    input_data["Credit_Score_sq"] = credit_score ** 2

    # One-hot encode categorical features
    categorical_columns = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    encoded_array = encoder.transform(
        input_data[categorical_columns]
    )

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(),
        index=input_data.index
    )

    # Select numeric features
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

    # Combine numerical and encoded features
    final_input = pd.concat(
        [numeric_df, encoded_df],
        axis=1
    )

    # Match the exact training feature order
    final_input = final_input.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale input for prediction
    scaled_input = scaler.transform(final_input)

    # Generate predictions from all three models
    predictions = {
        "Logistic Regression":
            logistic_model.predict(scaled_input)[0],

        "K-Nearest Neighbors":
            knn_model.predict(scaled_input)[0],

        "Gaussian Naive Bayes":
            nb_model.predict(scaled_input)[0]
    }

    # ---------------- DISPLAY RESULTS ----------------
    st.write("")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">◈ Prediction Results</div>
        <div class="panel-subtitle">
            Compare the output generated by each trained model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    result_cols = st.columns(3)

    for col, (model_name, prediction) in zip(
        result_cols, predictions.items()
    ):
        with col:

            if prediction == 1:
                status = "Loan Approved"
                status_class = "approved"
                symbol = "✓"
            else:
                status = "Loan Not Approved"
                status_class = "rejected"
                symbol = "×"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-model">{model_name}</div>
                <div class="result-status {status_class}">
                    {symbol} {status}
                </div>
                <div class="panel-subtitle">
                    Individual model prediction
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.caption(
        "For educational and demonstration purposes only. "
        "These predictions are not a guarantee of actual loan approval."
    )


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    CREDITWISE · MACHINE LEARNING DEMO<br>
    Built with Python · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
