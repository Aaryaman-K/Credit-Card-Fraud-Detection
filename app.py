import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained Decision Tree model
model = joblib.load('fraud_model.pkl')

st.title("Credit Card Fraud Detection System")

# Create tabs to organize the dashboard
tab1, tab2, tab3 = st.tabs(["Live Prediction", "Model Insights", "Data Explorer"])

with tab1:
    st.write("Enter the transaction details below to predict the likelihood of fraud.")
    
    # Input fields
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
        input_data = pd.DataFrame([[distance_home, distance_last, ratio_median, repeat_retailer, used_chip, used_pin, online_order]],
                                  columns=['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 
                                           'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order'])
        
        prediction = model.predict(input_data)
        
        if prediction[0] == 1.0:
            st.error("🚨 Alert: Fraudulent Transaction Detected!")
        else:
            st.success("✅ Transaction is Legitimate.")

with tab2:
    st.subheader("Decision Tree Feature Importance")
    st.write("This chart displays which factors drive the model's fraud predictions.")
    
    # Extract feature importance directly from the loaded model
    importances = model.feature_importances_
    features = ['Distance from Home', 'Distance from Last Txn', 'Ratio to Median Price', 
                'Repeat Retailer', 'Used Chip', 'Used PIN', 'Online Order']
    
    # Create and display the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importances, y=features, ax=ax, palette="Blues_r")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

with tab3:
    st.subheader("Correlation Heatmap")
    st.write("Visualizing how features correlate with fraudulent activity.")
    
    try:
        # Requires the dataset to be in the same directory
        df = pd.read_csv('card_transdata.csv')
        
        # Generate correlation matrix and heatmap
        corr = df.corr(numeric_only=True)
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="rocket", ax=ax2) #
        st.pyplot(fig2)
        
    except FileNotFoundError:
        st.warning("Please upload 'card_transdata.csv' to the repository to view the heatmap.")