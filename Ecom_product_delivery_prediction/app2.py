import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load the trained XGBoost model
try:
    with open("cxgboost_model.pkl", "rb") as file:
        model = pickle.load(file)  # Load using pickle
    print("Model loaded successfully from Pickle format!")
except Exception:
    model = xgb.Booster()
    model.load_model("xgboost_model_fixed.json")  # Load if using XGBoost native format
    print("Model loaded successfully from XGBoost JSON format!")

# Define the expected feature names (10 from dataset + 1 assumed)
feature_names = [
    'Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls', 'Customer_rating',
    'Cost_of_the_Product', 'Prior_purchases', 'Product_importance', 'Gender',
    'Discount_offered', 'Weight_in_gms'
]

# Preprocessing function
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Encode categorical variables
    warehouse_block_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
    mode_of_shipment_map = {'Flight': 0, 'Road': 1, 'Ship': 2}
    product_importance_map = {'low': 0, 'medium': 1, 'high': 2}
    gender_map = {'F': 0, 'M': 1}

    df['Warehouse_block'] = df['Warehouse_block'].map(warehouse_block_map)
    df['Mode_of_Shipment'] = df['Mode_of_Shipment'].map(mode_of_shipment_map)
    df['Product_importance'] = df['Product_importance'].map(product_importance_map)
    df['Gender'] = df['Gender'].map(gender_map)

    # StandardScaler (should match training)
    scaler = StandardScaler()
    numerical_cols = ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
                      'Prior_purchases', 'Discount_offered', 'Weight_in_gms']
    
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Ensure feature order matches model training
    df = df[feature_names]

    # If model expects an extra feature, add it as a constant
    df['Dummy_feature'] = 0  # Ensure alignment if needed

    return df

import numpy as np
import xgboost as xgb

def predict_delivery(input_data):
    processed_data = preprocess_input(input_data)

    # Convert DataFrame to NumPy array (needed for XGBClassifier)
    processed_array = np.array(processed_data)

    # Check if model is an XGBClassifier (Scikit-Learn API) or Booster (XGBoost's native model)
    if isinstance(model, xgb.XGBClassifier):  
        prediction = model.predict(processed_array)  # Use NumPy array for XGBClassifier
    elif isinstance(model, xgb.Booster):
        dmatrix_data = xgb.DMatrix(processed_data)  # Convert to DMatrix for Booster models
        prediction = model.predict(dmatrix_data)
    else:
        raise ValueError("Unknown model type. Make sure you're using an XGBoost model.")

    return int(prediction[0])  # Convert to integer (0 or 1)


# Streamlit UI
st.set_page_config(page_title="E-Commerce Delivery Prediction", page_icon="📦", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
    .title { font-size: 2.5em; color: #2e8b57; text-align: center; font-weight: bold; }
    .subtitle { font-size: 1.2em; color: #4a90e2; text-align: center; }
    .input-box { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
    .predict-button { background-color: #1abc9c; color: white; padding: 10px 20px; border-radius: 5px; font-size: 1.1em; font-weight: bold; }
    .predict-button:hover { background-color: #16a085; }
    .result-success { color: #2ecc71; font-size: 1.5em; font-weight: bold; text-align: center; }
    .result-failure { color: #e74c3c; font-size: 1.5em; font-weight: bold; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Main layout
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<div class='title'>E-Commerce Product Delivery Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether your product will be delivered on time!</div>", unsafe_allow_html=True)

# Input form
with st.form(key='delivery_form'):
    st.markdown("<div class='input-box'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        warehouse_block = st.selectbox("Warehouse Block", options=['A', 'B', 'C', 'D', 'F'])
        mode_of_shipment = st.selectbox("Mode of Shipment", options=['Flight', 'Road', 'Ship'])
        customer_care_calls = st.slider("Customer Care Calls", min_value=1, max_value=10, value=4)
        customer_rating = st.slider("Customer Rating", min_value=1, max_value=5, value=3)
        cost_of_product = st.number_input("Cost of Product ($)", min_value=50, max_value=500, value=200)

    with col2:
        prior_purchases = st.slider("Prior Purchases", min_value=1, max_value=10, value=3)
        product_importance = st.selectbox("Product Importance", options=['low', 'medium', 'high'])
        gender = st.selectbox("Gender", options=['F', 'M'])
        discount_offered = st.number_input("Discount Offered (%)", min_value=0, max_value=100, value=10)
        weight_in_gms = st.number_input("Weight (grams)", min_value=500, max_value=8000, value=2000)

    st.markdown("</div>", unsafe_allow_html=True)

    submit_button = st.form_submit_button(label="Predict Delivery", help="Click to predict", type="primary")

# Prediction logic
if submit_button:
    input_data = {
        'Warehouse_block': warehouse_block,
        'Mode_of_Shipment': mode_of_shipment,
        'Customer_care_calls': customer_care_calls,
        'Customer_rating': customer_rating,
        'Cost_of_the_Product': cost_of_product,
        'Prior_purchases': prior_purchases,
        'Product_importance': product_importance,
        'Gender': gender,
        'Discount_offered': discount_offered,
        'Weight_in_gms': weight_in_gms
    }

    try:
        prediction = predict_delivery(input_data)
        if prediction == 0:
            st.markdown("<div class='result-success'>✅ Product is predicted to be delivered ON TIME!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-failure'>❌ Product is predicted to be DELAYED!</div>", unsafe_allow_html=True)
    except ValueError as e:
        st.error(f"Error: {e}. Please ensure all inputs match the model's expected features.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #7f8c8d;'>Powered by xAI | Built with Streamlit</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
