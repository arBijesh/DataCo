import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64

# ✅ First Streamlit command (MUST be at the very top)
st.set_page_config(page_title="Profit Predictor - DataCo Supply Chain", page_icon="📦", layout="wide")

# Function to encode the local image
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Apply background image with opacity effect
def set_background(image_path, opacity=0.4):  # Adjust opacity (0.0 - fully transparent, 1.0 - fully visible)
    base64_image = get_base64_of_image(image_path)
    bg_style = f"""
    <style>
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})), 
                    url("data:image/png;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        z-index: -1;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);  /* Transparent header */
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# Call the function to set background (make sure "background.jpg" exists in the same folder)
set_background("background.jpg", opacity=0.5)  # Adjust opacity here

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

st.title("Profit Predictor: Enhancing Business Decisions with Data Science")
st.markdown("### 📊 Enter Order Details to Predict Profit:")

# Dropdowns
selected_market = st.selectbox("Market", market_options)
selected_department = st.selectbox("Department Name", department_options, key="department")

# Automatically set the profit ratio based on department selection
order_item_profit_ratio = profit_ratio_defaults.get(selected_department, 0.0)
order_item_profit_ratio = st.number_input("Order Item Profit Ratio", min_value=0.0, value=order_item_profit_ratio, step=0.01)

# Numeric inputs
order_item_total = st.number_input("Order Item Total", min_value=0.0, value=500.0, step=10.0)
sales_per_customer = st.number_input("Sales per customer", min_value=0.0, value=100.0, step=1.0)
order_item_product_price = st.number_input("Order Item Product Price", min_value=0.0, value=50.0, step=1.0)
sales = st.number_input("Sales", min_value=0.0, value=1000.0, step=10.0)
product_price = st.number_input("Product Price", min_value=0.0, value=200.0, step=1.0)

# Create one-hot encoded columns
input_data = {col: 0 for col in expected_columns}  # Initialize all to 0

# Set numerical values
input_data.update({
    "Order Item Profit Ratio": order_item_profit_ratio,
    "Order Item Total": order_item_total,
    "Sales per customer": sales_per_customer,
    "Order Item Product Price": order_item_product_price,
    "Sales": sales,
    "Product Price": product_price
})

# Set one-hot encoding for selected market
input_data[f"Market_{selected_market}"] = 1

# Set one-hot encoding for selected department (keeping space as in trained model)
input_data[f"Department Name_{selected_department} "] = 1  # Notice the space!

# Convert to DataFrame
input_df = pd.DataFrame([input_data])[expected_columns]

# Predict when button is pressed
if st.button("Predict"):
    prediction = gb_model_final.predict(input_df)
    
    # Display the prediction
    st.subheader("Predicted Profit (in USD):")
    st.markdown(f"### 💲 **${prediction[0]:.2f}**")
    st.success("The model has successfully made a prediction!")
