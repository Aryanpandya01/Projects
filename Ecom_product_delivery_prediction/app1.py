import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained XGBoost model
with open("bxgboost_model.pkl.pkl", "rb") as file:
    model = pickle.load(file)

# Define the input features based on dataset columns
feature_names = [
    "Warehouse_block", "Mode_of_Shipment", "Customer_care_calls",
    "Customer_rating", "Cost_of_the_Product", "Prior_purchases",
    "Product_importance", "Gender", "Discount_offered", "Weight_in_gms"
]

def user_input_features():
    st.sidebar.header("Enter the details for prediction")
    
    warehouse_block = st.sidebar.selectbox("Warehouse Block", ["A", "B", "C", "D", "E"])
    mode_of_shipment = st.sidebar.selectbox("Mode of Shipment", ["Ship", "Flight", "Road"])
    customer_care_calls = st.sidebar.slider("Customer Care Calls", 0, 10, 2)
    customer_rating = st.sidebar.slider("Customer Rating", 1, 5, 3)
    cost_of_product = st.sidebar.number_input("Cost of Product", min_value=10, max_value=500, value=100)
    prior_purchases = st.sidebar.slider("Prior Purchases", 0, 10, 2)
    product_importance = st.sidebar.selectbox("Product Importance", ["low", "medium", "high"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    discount_offered = st.sidebar.slider("Discount Offered", 0, 50, 5)
    weight_in_gms = st.sidebar.number_input("Weight in grams", min_value=100, max_value=5000, value=1000)
    
    data = pd.DataFrame({
        "Warehouse_block": [warehouse_block],
        "Mode_of_Shipment": [mode_of_shipment],
        "Customer_care_calls": [customer_care_calls],
        "Customer_rating": [customer_rating],
        "Cost_of_the_Product": [cost_of_product],
        "Prior_purchases": [prior_purchases],
        "Product_importance": [product_importance],
        "Gender": [gender],
        "Discount_offered": [discount_offered],
        "Weight_in_gms": [weight_in_gms]
    })
    return data

st.title("E-Commerce Product Delivery Prediction")
st.write("This app predicts whether a product will be delivered on time or not based on user inputs.")

input_df = user_input_features()

# One-hot encode categorical variables
input_df = pd.get_dummies(input_df)

# Ensure input columns match model training columns
if hasattr(model, 'feature_names_in_'):
    model_features = model.feature_names_in_
else:
    model_features = input_df.columns  # Fallback if feature names are unavailable

for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with default value

input_df = input_df[model_features]  # Arrange columns in the same order

# Predict
if st.button("Predict Delivery Status"):
    prediction = model.predict(input_df)
    result = "On Time" if prediction[0] == 1 else "Delayed"
    st.write(f"### Prediction: {result}")
