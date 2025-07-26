###### -----streamlit app test1 ----- ######
import streamlit as st
import random
import pandas as pd
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Churn Prediction", layout="wide")

# --- Custom Title ---
st.markdown("""
<div style='text-align: center;'>
    <h1>🐧 Churn Prediction Stream 🐧</h1>
    <p style='font-size: 18px; color: gray;'>by Hassan Ahmed</p>
    <p style='font-size: 14px; color: gray;'>This is a Streamlit app for predicting customer churn in a telecom company.</p>
    <p style='font-size: 14px; color: gray;'>You can input customer data and get a prediction on whether the customer is likely to churn.</p>
    <p style='font-size: 14px; color: gray;'>The app uses a machine learning model trained on customer data.</p>
    <p style='font-size: 14px; color: gray;'>Feel free to explore the features and make predictions!</p>
    <p style='font-size: 14px; color: gray;'>This app is built using Streamlit.</p>
    <p style='font-size: 14px; color: gray;'>Enjoy using the app!</p>
</div>
""", unsafe_allow_html=True)

# --- Mappings ---
gender_map = {1: 'Female', 0: 'Male'}
partner_map = {1: 'Yes', 0: 'No'}
dependents_map = {1: 'Yes', 0: 'No'}
phone_map = {1: 'Yes', 0: 'No'}
multiple_map = {1: 'Yes', 0: 'No'}
internet_map = {0: 'DSL', 1: 'Fiber optic', 2: 'No'}
online_map = {1: 'Yes', 0: 'No'}
contract_map = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
paperless_map = {1: 'Yes', 0: 'No'}
payment_map = {0: 'Electronic check', 1: 'Mailed check', 2: 'Bank transfer (automatic)', 3: 'Credit card (automatic)'}

def auto_generate():
    for key, value in {
        'gender': random.choice([1, 0]),
        'SeniorCitizen': random.choice([0, 1]),
        'Partner': random.choice([1, 0]),
        'Dependents': random.choice([1, 0]),
        'tenure': random.randint(0, 72),
        'PhoneService': random.choice([1, 0]),
        'MultipleLines': random.choice([1, 0]),
        'InternetService': random.choice([0, 1, 2]),
        'OnlineSecurity': random.choice([1, 0]),
        'OnlineBackup': random.choice([1, 0]),
        'DeviceProtection': random.choice([1, 0]),
        'TechSupport': random.choice([1, 0]),
        'StreamingTV': random.choice([1, 0]),
        'StreamingMovies': random.choice([1, 0]),
        'Contract': random.choice([0, 1, 2]),
        'PaperlessBilling': random.choice([1, 0]),
        'PaymentMethod': random.choice([0, 1, 2, 3]),
        'MonthlyCharges': round(random.uniform(0.0, 120.0), 2),
        'TotalCharges': round(random.uniform(0.0, 10000.0), 2)
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Tabbed Layout ---
tab1, tab2 = st.tabs(["📈 Churn Predictor", "📝 History Log"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = st.selectbox("Gender", options=list(gender_map.keys()), format_func=lambda x: gender_map[x], key='gender')
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], key='SeniorCitizen')
        Partner = st.selectbox("Partner", options=list(partner_map.keys()), format_func=lambda x: partner_map[x], key='Partner')
        Dependents = st.selectbox("Dependents", options=list(dependents_map.keys()), format_func=lambda x: dependents_map[x], key='Dependents')
        tenure = st.slider("Tenure (Months)", 0, 72, value=st.session_state.get('tenure', 12), key='tenure')
        tenure = st.number_input("Edit Tenure", min_value=0, max_value=72, value=tenure, step=1, key='tenure_input')

    with col2:
        OnlineSecurity = st.selectbox("Online Security", options=list(online_map.keys()), format_func=lambda x: online_map[x], key='OnlineSecurity')
        OnlineBackup = st.selectbox("Online Backup", options=list(online_map.keys()), format_func=lambda x: online_map[x], key='OnlineBackup')
        DeviceProtection = st.selectbox("Device Protection", options=list(online_map.keys()), format_func=lambda x: online_map[x], key='DeviceProtection')
        TechSupport = st.selectbox("Tech Support", options=list(online_map.keys()), format_func=lambda x: online_map[x], key='TechSupport')
        MonthlyCharges = st.slider("Monthly Charges", 0.0, 120.0, value=st.session_state.get('MonthlyCharges', 70.0), key='MonthlyCharges')
        MonthlyCharges = st.number_input("Edit Monthly Charges", min_value=0.0, max_value=120.0, value=MonthlyCharges, step=0.1, key='MonthlyCharges_input')

    with col3:
        StreamingTV = st.selectbox("Streaming TV", options=list(online_map.keys()), format_func=lambda x: online_map[x], key='StreamingTV')
        StreamingMovies = st.selectbox("Streaming Movies", options=list(online_map.keys()), format_func=lambda x: online_map[x], key='StreamingMovies')
        MultipleLines = st.selectbox("Multiple Lines", options=list(multiple_map.keys()), format_func=lambda x: multiple_map[x], key='MultipleLines')
        PhoneService = st.selectbox("Phone Service", options=list(phone_map.keys()), format_func=lambda x: phone_map[x], key='PhoneService')
        TotalCharges = st.slider("Total Charges", 0.0, 10000.0, value=st.session_state.get('TotalCharges', 350.0), key='TotalCharges')
        TotalCharges = st.number_input("Edit Total Charges", min_value=0.0, max_value=10000.0, value=TotalCharges, step=1.0, key='TotalCharges_input')

    with col4:
        Contract = st.selectbox("Contract", options=list(contract_map.keys()), format_func=lambda x: contract_map[x], key='Contract')
        PaperlessBilling = st.selectbox("Paperless Billing", options=list(paperless_map.keys()), format_func=lambda x: paperless_map[x], key='PaperlessBilling')
        PaymentMethod = st.selectbox("Payment Method", options=list(payment_map.keys()), format_func=lambda x: payment_map[x], key='PaymentMethod')
        InternetService = st.selectbox("Internet Service", options=list(internet_map.keys()), format_func=lambda x: internet_map[x], key='InternetService')

    # --- Decoded Display ---
    decoded_data = {
        'Gender': gender_map[gender],
        'SeniorCitizen': 'Yes' if SeniorCitizen == 1 else 'No',
        'Partner': partner_map[Partner],
        'Dependents': dependents_map[Dependents],
        'Tenure': tenure,
        'PhoneService': phone_map[PhoneService],
        'MultipleLines': multiple_map[MultipleLines],
        'InternetService': internet_map[InternetService],
        'OnlineSecurity': online_map[OnlineSecurity],
        'OnlineBackup': online_map[OnlineBackup],
        'DeviceProtection': online_map[DeviceProtection],
        'TechSupport': online_map[TechSupport],
        'StreamingTV': online_map[StreamingTV],
        'StreamingMovies': online_map[StreamingMovies],
        'Contract': contract_map[Contract],
        'PaperlessBilling': paperless_map[PaperlessBilling],
        'PaymentMethod': payment_map[PaymentMethod],
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    st.markdown("### Current Data (Readable)")
    st.dataframe(pd.DataFrame([decoded_data]))

    # --- Encoded for Prediction ---
    encoded_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    colA, colB = st.columns(2)

    with colA:
        if st.button("Random Generate Data"):
            for key in [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'TotalCharges'
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            auto_generate()
            st.rerun()

with colB:
    if st.button("Predict Churn"):
        import joblib

        # Load model
        model = joblib.load("streamed_churn/best_telco_model.pkl")
        # Make prediction
        pred = model.predict(encoded_data)[0]
        prob = model.predict_proba(encoded_data)[0][1]
        result_label = "Churn" if pred == 1 else "No Churn"

        st.success(f"Prediction: {result_label} (Probability: {prob:.2f})")

        # Store history in session
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []

        # A row with readable input data and result
        log_row = decoded_data.copy()
        log_row["Result"] = result_label
        log_row["Probability"] = f"{prob:.2f}"

        st.session_state.prediction_history.append(log_row)
with tab2:
    st.markdown("### 📝 Prediction History")

    if "prediction_history" in st.session_state and st.session_state.prediction_history:
        hist_df = pd.DataFrame(st.session_state.prediction_history)

        # Add a colored churn status column
        def color_churn(val):
            if val == "Churn":
                return f"<span style='color: red; font-weight: bold;'>Churn</span>"
            else:
                return f"<span style='color: green; font-weight: bold;'>No Churn</span>"

        hist_df["Churn Status"] = hist_df["Result"].apply(color_churn)

        # Reorder columns 
        cols = ["Churn Status"] + [c for c in hist_df.columns if c not in ["Churn Status", "Result"]]
        hist_df = hist_df[cols]

        st.write(hist_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("No predictions made yet.")
