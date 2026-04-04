import streamlit as st
import pandas as pd
import joblib

model  = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("Customer Churn Predictor 🔮")
st.write("Enter customer details to predict if they will churn")

col1, col2 = st.columns(2)

with col1:
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    senior_citizen  = st.selectbox("Senior Citizen", [0, 1])
    partner         = st.selectbox("Partner", ["Yes", "No"])
    dependents      = st.selectbox("Dependents", ["Yes", "No"])
    phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])

with col2:
    contract        = st.selectbox("Contract",
                        ["Month-to-month", "One year", "Two year"])
    internet        = st.selectbox("Internet Service",
                        ["DSL", "Fiber optic", "No"])
    payment         = st.selectbox("Payment Method",
                        ["Electronic check", "Mailed check",
                         "Bank transfer (automatic)",
                         "Credit card (automatic)"])
    multiple_lines  = st.selectbox("Multiple Lines",
                        ["Yes", "No", "No phone service"])
    online_security = st.selectbox("Online Security",
                        ["Yes", "No", "No internet service"])
    online_backup   = st.selectbox("Online Backup",
                        ["Yes", "No", "No internet service"])
    tech_support    = st.selectbox("Tech Support",
                        ["Yes", "No", "No internet service"])

if st.button("Predict Churn", type="primary"):

    total_charges = tenure * monthly_charges

    # Exact column order matching scaler.feature_names_in_
    input_dict = {
        'gender':           1,
        'SeniorCitizen':    senior_citizen,
        'Partner':          1 if partner == "Yes" else 0,
        'Dependents':       1 if dependents == "Yes" else 0,
        'tenure':           tenure,
        'PhoneService':     1 if phone_service == "Yes" else 0,
        'MultipleLines':    1 if multiple_lines == "Yes" else 0,
        'OnlineSecurity':   1 if online_security == "Yes" else 0,
        'OnlineBackup':     1 if online_backup == "Yes" else 0,
        'DeviceProtection': 0,
        'TechSupport':      1 if tech_support == "Yes" else 0,
        'StreamingTV':      0,
        'StreamingMovies':  0,
        'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'MonthlyCharges':   monthly_charges,
        'TotalCharges':     total_charges,
        'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
        'InternetService_No':          1 if internet == "No" else 0,
        'Contract_One year':           1 if contract == "One year" else 0,
        'Contract_Two year':           1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check':        1 if payment == "Electronic check" else 0,
        'PaymentMethod_Mailed check':            1 if payment == "Mailed check" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — {probability:.0%} probability of leaving")
        st.write("**Recommendation:** Offer this customer a discount or upgrade")
    else:
        st.success(f"✅ Low Churn Risk — {1-probability:.0%} probability of staying")
        st.write("**Recommendation:** Customer is likely to stay")

    st.metric("Churn Probability", f"{probability:.0%}")
    st.progress(float(probability))
