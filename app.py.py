import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set up the page early to avoid Streamlit errors
st.set_page_config(page_title="Profit Predictor - DataCo Supply Chain", page_icon="📦", layout="wide")

# Custom CSS for background and frosted-glass effect
st.markdown(
    """
    <style>
    body {
        background: url('https://source.unsplash.com/1600x900/?supply-chain,logistics') no-repeat center center fixed;
        background-size: cover;
    }
    .stApp {
        background: rgba(0, 0, 0, 0.4);
        padding: 20px;
        border-radius: 15px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white;
    }
    .stSelectbox, .stNumber_input, .stButton {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 10px;
        color: white;
        backdrop-filter: blur(10px);
    }
    .stButton > button {
        background: linear-gradient(45deg, #FF512F, #DD2476);
        color: white;
        font-size: 18px;
        border: none;
        padding: 12px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #DD2476, #FF512F);
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚀 Profit Predictor: Enhancing Business Decisions with Data Science")
st.markdown("### 📊 Enter Order Details to Predict Profit:")

# Load trained model
with open('gb_model_final_hyper.pkl', 'rb') as model_file:
    gb_model_final = pickle.load(model_file)

# Get expected feature names from the model
expected_columns = list(gb_model_final.feature_names_in_)

# Define categorical options
market_options = ["Europe", "LATAM", "Pacific Asia", "USCA"]
department_options = ["Book Shop", "Discs Shop", "Fan Shop", "Fitness", "Footwear",
                      "Golf", "Health and Beauty ", "Outdoors", "Pet Shop", "Technology"]

# Default Profit Ratios
profit_ratio_defaults = {
    "Book Shop": 0.46, "Discs Shop": 0.14, "Fan Shop": 0.13, "Fitness": 0.0059,
    "Footwear": 0.0043, "Golf": 0.0001, "Health and Beauty ": 0.0001, "Outdoors": 0.0005,
    "Pet Shop": 0.0001, "Technology": 0.0020
}

# UI Inputs
selected_market = st.selectbox("🌎 Market", market_options)
selected_department = st.selectbox("🏪 Department Name", department_options)
order_item_profit_ratio = profit_ratio_defaults.get(selected_department, 0.0)
order_item_profit_ratio = st.number_input("💰 Order Item Profit Ratio", min_value=0.0, value=order_item_profit_ratio, step=0.01)

# Numeric inputs
order_item_total = st.number_input("📦 Order Item Total", min_value=0.0, value=500.0, step=10.0)
sales_per_customer = st.number_input("👤 Sales per customer", min_value=0.0, value=100.0, step=1.0)
order_item_product_price = st.number_input("🏷 Order Item Product Price", min_value=0.0, value=50.0, step=1.0)
sales = st.number_input("📈 Sales", min_value=0.0, value=1000.0, step=10.0)
product_price = st.number_input("💲 Product Price", min_value=0.0, value=200.0, step=1.0)

# Data Processing for Model
input_data = {col: 0 for col in expected_columns}
input_data.update({
    "Order Item Profit Ratio": order_item_profit_ratio,
    "Order Item Total": order_item_total,
    "Sales per customer": sales_per_customer,
    "Order Item Product Price": order_item_product_price,
    "Sales": sales,
    "Product Price": product_price
})
input_data[f"Market_{selected_market}"] = 1
input_data[f"Department Name_{selected_department} "] = 1  # Notice the space!
input_df = pd.DataFrame([input_data])[expected_columns]

# Predict Button
if st.button("🚀 Predict Profit"):
    prediction = gb_model_final.predict(input_df)
    st.subheader("Predicted Profit (in USD):")
    st.markdown(f"### 💲 **${prediction[0]:.2f}**")
    st.success("✅ Prediction Successful!")
