import streamlit as st
import pandas as pd
import joblib

# --- Load saved model and scaler ---
model  = joblib.load('../models/model.pkl')
scaler = joblib.load('../models/scaler.pkl')

st.title("Customer Churn Predictor 🔮")
st.write("Enter customer details to predict if they will churn")

# --- User Input ---
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

# --- Predict Button ---
if st.button("Predict Churn", type="primary"):

    # Step 1: Calculate TotalCharges
    total_charges = tenure * monthly_charges

    # Step 2: Build raw dataframe
    input_dict = {
        'SeniorCitizen':    senior_citizen,
        'tenure':           tenure,
        'MonthlyCharges':   monthly_charges,
        'TotalCharges':     total_charges,
        'gender':           'Male',       # default
        'Partner':          partner,
        'Dependents':       dependents,
        'PhoneService':     phone_service,
        'MultipleLines':    multiple_lines,
        'OnlineSecurity':   online_security,
        'OnlineBackup':     online_backup,
        'DeviceProtection': 'No',         # default
        'TechSupport':      tech_support,
        'StreamingTV':      'No',         # default
        'StreamingMovies':  'No',         # default
        'PaperlessBilling': paperless,
        'InternetService':  internet,
        'Contract':         contract,
        'PaymentMethod':    payment
    }
    input_df = pd.DataFrame([input_dict])

    # Step 3: Label encode binary columns
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0,
                  'No phone service': 0, 'No internet service': 0}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'MultipleLines', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies']
    for col in binary_cols:
        input_df[col] = input_df[col].map(binary_map)

    # Step 4: One Hot Encode multi-category columns
    input_df = pd.get_dummies(input_df,
                   columns=['InternetService', 'Contract', 'PaymentMethod'])

    # Step 5: Align columns with training data
    # Add any missing columns with 0
    train_cols = model.get_booster().feature_names
    for col in train_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[train_cols]  # reorder to match training

    # Step 6: Scale using saved scaler
    input_scaled = scaler.transform(input_df)

    # Step 7: Predict
    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Step 8: Show result
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — {probability:.0%} probability of leaving")
        st.write("**Recommendation:** Offer this customer a discount or upgrade")
    else:
        st.success(f"✅ Low Churn Risk — {1-probability:.0%} probability of staying")
        st.write("**Recommendation:** Customer is likely to stay")

    # Show probability bar
    st.metric("Churn Probability", f"{probability:.0%}")
    st.progress(float(probability))