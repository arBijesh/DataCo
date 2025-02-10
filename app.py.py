import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('gb_model_final.pkl', 'rb') as model_file:
    gb_model_final = pickle.load(model_file)

# Define Market and Department categories (MUST match training data)
market_options = ["Europe", "LATAM", "Pacific Asia", "USCA"]
department_options = ["Book Shop", "Discs Shop", "Fan Shop", "Fitness", "Footwear",
                      "Golf", "Health and Beauty", "Outdoors", "Pet Shop", "Technology"]

# Default Order Item Profit Ratio based on Department Name
profit_ratio_defaults = {
    "Book Shop": 0.25,
    "Discs Shop": 0.30,
    "Fan Shop": 0.35,
    "Fitness": 0.40,
    "Footwear": 0.45,
    "Golf": 0.50,
    "Health and Beauty": 0.55,
    "Outdoors": 0.60,
    "Pet Shop": 0.65,
    "Technology": 0.70
}

# Streamlit UI setup
st.set_page_config(page_title="Profit Predictor - DataCo Supply Chain", page_icon="📦", layout="wide")

# Title and Description
st.title("Profit Predictor: Enhancing Business Decisions with Data Science")

# Dataset Introduction
st.markdown("""
    ### About the Dataset
    The **DataCo Global Supply Chain Dataset** provides insights into key **supply chain** metrics.
    This predictor helps businesses optimize **profitability** by leveraging Machine Learning.
""")

# User Inputs for Numerical Features
st.markdown("### 📊 Enter Order Details to Predict Profit:")

# Dropdowns for categorical variables
selected_market = st.selectbox("Market", market_options)
selected_department = st.selectbox("Department Name", department_options, key="department")

# Automatically set the profit ratio based on department selection
if "profit_ratio" not in st.session_state:
    st.session_state.profit_ratio = profit_ratio_defaults[selected_department]

if selected_department in profit_ratio_defaults:
    st.session_state.profit_ratio = profit_ratio_defaults[selected_department]

order_item_profit_ratio = st.number_input(
    "Order Item Profit Ratio", 
    min_value=0.0, 
    value=st.session_state.profit_ratio, 
    step=0.01
)

sales_per_customer = st.number_input("Sales per Customer", min_value=0.0, value=100.0, step=1.0)
order_item_total = st.number_input("Order Item Total", min_value=0.0, value=500.0, step=10.0)
order_item_product_price = st.number_input("Order Item Product Price", min_value=0.0, value=50.0, step=1.0)
sales = st.number_input("Sales", min_value=0.0, value=1000.0, step=10.0)
product_price = st.number_input("Product Price", min_value=0.0, value=200.0, step=1.0)

# Convert categorical inputs to one-hot encoding
market_encoded = {f"Market_{m}": (1 if m == selected_market else 0) for m in market_options}
department_encoded = {f"Department Name_{d}": (1 if d == selected_department else 0) for d in department_options}

# Combine all input data into a DataFrame
input_data = pd.DataFrame([{
    "Order Item Profit Ratio": order_item_profit_ratio,
    "Sales per customer": sales_per_customer,
    "Order Item Total": order_item_total,
    "Order Item Product Price": order_item_product_price,
    "Sales": sales,
    "Product Price": product_price,
    **market_encoded,
    **department_encoded
}])

# Ensure input columns match model training columns (if extra columns exist, they will be ignored)
expected_columns = [
    "Order Item Profit Ratio", "Sales per customer", "Order Item Total", "Order Item Product Price",
    "Sales", "Product Price",
    "Market_Europe", "Market_LATAM", "Market_Pacific Asia", "Market_USCA",
    "Department Name_Book Shop", "Department Name_Discs Shop", "Department Name_Fan Shop",
    "Department Name_Fitness", "Department Name_Footwear", "Department Name_Golf",
    "Department Name_Health and Beauty", "Department Name_Outdoors", "Department Name_Pet Shop",
    "Department Name_Technology"
]

# Add missing columns with 0 (important for consistency with training data)
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match training data
input_data = input_data[expected_columns]

# Make Prediction
if st.button("Predict"):
    # Predict using the trained model
    prediction = gb_model_final.predict(input_data)
    
    # Display the prediction with a formatted dollar amount
    st.subheader("Predicted Profit per Order (in USD):")
    st.markdown(f"### 💲 **${prediction[0]:.2f}**")

    # Add a success message for better user experience
    st.success("Prediction generated successfully!")
