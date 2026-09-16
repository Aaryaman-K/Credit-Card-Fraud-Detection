import streamlit as st
import pandas as pd
import joblib

# Load the trained Decision Tree model
model = joblib.load('fraud_model.pkl')

st.title("Credit Card Fraud Detection System")
st.write("Enter the transaction details below to predict the likelihood of fraud.")

# Create input fields for the 7 features
distance_home = st.number_input("Distance from Home", min_value=0.0, value=10.0)
distance_last = st.number_input("Distance from Last Transaction", min_value=0.0, value=1.0)
ratio_median = st.number_input("Ratio to Median Purchase Price", min_value=0.0, value=1.0)

col1, col2 = st.columns(2)
with col1:
    repeat_retailer = st.selectbox("Repeat Retailer", [0.0, 1.0])
    used_chip = st.selectbox("Used Chip", [0.0, 1.0])
with col2:
    used_pin = st.selectbox("Used PIN Number", [0.0, 1.0])
    online_order = st.selectbox("Online Order", [0.0, 1.0])

if st.button("Analyze Transaction"):
    # Format the inputs into a Pandas DataFrame
    input_data = pd.DataFrame([[distance_home, distance_last, ratio_median, repeat_retailer, used_chip, used_pin, online_order]],
                              columns=['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 
                                       'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order'])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    if prediction[0] == 1.0:
        st.error("🚨 Alert: Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")