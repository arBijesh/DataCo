import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open('gb_model_final_hyper.pkl', 'rb') as model_file:
    gb_model_final = pickle.load(model_file)

# Feature names from training data
expected_columns = [
    "Order Item Profit Ratio", "Order Item Total", "Sales per customer", "Product Price",
    "Order Item Product Price", "Sales",
    "Department Name_Technology", "Market_Europe", "Department Name_Footwear",
    "Market_Pacific Asia", "Department Name_Outdoors", "Market_LATAM", "Market_USCA",
    "Department Name_Discs Shop", "Department Name_Fan Shop", "Department Name_Fitness",
    "Department Name_Golf", "Department Name_Health and Beauty", "Department Name_Pet Shop",
    "Department Name_Book Shop"
]

# Categorical options (matching model training)
market_options = ["Europe", "LATAM", "Pacific Asia", "USCA"]
department_options = ["Book Shop", "Discs Shop", "Fan Shop", "Fitness", "Footwear",
                      "Golf", "Health and Beauty", "Outdoors", "Pet Shop", "Technology"]

# Default Order Item Profit Ratio (all below 0.47)
profit_ratio_defaults = {
    "Book Shop": 0.46, "Discs Shop": 0.14, "Fan Shop": 0.13, "Fitness": 0.0059,
    "Footwear": 0.0043, "Golf": 0.0001, "Health and Beauty": 0.0001, "Outdoors": 0.0005,
    "Pet Shop": 0.0001, "Technology": 0.0020
}

# Streamlit UI setup
st.set_page_config(page_title="Profit Predictor - DataCo Supply Chain", page_icon="📦", layout="wide")

st.title("Profit Predictor: Enhancing Business Decisions with Data Science")

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

# Numeric inputs
order_item_total = st.number_input("Order Item Total", min_value=0.0, value=500.0, step=10.0)
sales_per_customer = st.number_input("Sales per customer", min_value=0.0, value=100.0, step=1.0)
order_item_product_price = st.number_input("Order Item Product Price", min_value=0.0, value=50.0, step=1.0)
sales = st.number_input("Sales", min_value=0.0, value=1000.0, step=10.0)
product_price = st.number_input("Product Price", min_value=0.0, value=200.0, step=1.0)

# One-hot encode categorical variables (ensuring names match training data)
market_encoded = {f"Market_{m}": (1 if m == selected_market else 0) for m in market_options}
department_encoded = {f"Department Name_{d}": (1 if d == selected_department else 0) for d in department_options}

# Create input DataFrame
input_data = pd.DataFrame([{
    "Order Item Profit Ratio": order_item_profit_ratio,
    "Order Item Total": order_item_total,
    "Sales per customer": sales_per_customer,
    "Product Price": product_price,
    "Order Item Product Price": order_item_product_price,
    "Sales": sales,
    **department_encoded,
    **market_encoded
}])

# Ensure input matches expected columns (add missing columns with 0)
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0  # Ensures missing features are accounted for

# Reorder columns to match training order
input_data = input_data[expected_columns]

# Make Prediction
if st.button("Predict"):
    # Debugging info
    st.write("🔍 **Model expects features:**", gb_model_final.feature_names_in_)
    st.write("📊 **Current input features:**", list(input_data.columns))

    # Predict
    prediction = gb_model_final.predict(input_data)
    
    # Display output
    st.subheader("Predicted Profit per Order (in USD):")
    st.markdown(f"### 💲 **${prediction[0]:.2f}**")
    st.success("Prediction generated successfully!")
