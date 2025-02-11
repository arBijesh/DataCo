import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set up Streamlit page
st.set_page_config(page_title="Profit Predictor - DataCo Supply Chain", page_icon="📦", layout="wide")

st.title("🚀 Profit Predictor: Enhancing Business Decisions with Data Science")
st.markdown("### 📊 Enter Order Details to Predict Profit:")

# Load trained model
with open("gb_model_final_hyperparameter.pkl", "rb") as model_file:
    gb_model_final = pickle.load(model_file)

# Load label encoders
market_encoder = pickle.load(open("market_encoder.pkl", "rb"))
order_region_encoder = pickle.load(open("order_region_encoder.pkl", "rb"))
order_country_encoder = pickle.load(open("order_country_encoder.pkl", "rb"))

# Get expected feature names
expected_columns = list(gb_model_final.feature_names_in_)

# Define categorical options
market_options = ["Europe", "LATAM", "Pacific Asia", "USCA"]
order_region_options = ['Southeast Asia', 'South Asia', 'Oceania', 'Eastern Asia',
       'West Asia', 'West of USA ', 'US Center ', 'West Africa',
       'Central Africa', 'North Africa', 'Western Europe',
       'Northern Europe', 'Central America', 'Caribbean', 'South America',
       'East Africa', 'Southern Europe', 'East of USA', 'Canada',
       'Southern Africa', 'Central Asia', 'Eastern Europe',
       'South of  USA ']
department_options = ["Book Shop", "Discs Shop", "Fan Shop", "Fitness", "Footwear",
                      "Golf", "Health and Beauty", "Outdoors", "Pet Shop", "Technology"]

# Default profit ratio for each department
default_profit_ratios = {
    "Book Shop": 0.10,
    "Discs Shop": 0.08,
    "Fan Shop": 0.12,
    "Fitness": 0.15,
    "Footwear": 0.18,
    "Golf": 0.20,
    "Health and Beauty": 0.25,
    "Outdoors": 0.22,
    "Pet Shop": 0.17,
    "Technology": 0.30
}

# UI Inputs
selected_market = st.selectbox("🌎 Market", market_options)
selected_region = st.selectbox("🏙 Order Region", order_region_options)
selected_department = st.selectbox("🏮 Department Name", department_options)

# Assign default profit ratio based on department selection
default_profit_ratio = default_profit_ratios.get(selected_department, 0.10)

order_item_profit_ratio = st.number_input("💰 Order Item Profit Ratio", min_value=-1.0, value=default_profit_ratio, step=0.01)
product_price = st.number_input("💲 Product Price", min_value=0.0, value=200.0, step=1.0)
order_item_discount_rate = st.number_input("📆 Order Item Discount Rate", min_value=0.0, value=0.05, step=0.01)

# Convert categorical variables using label encoding
market_encoded = market_encoder.transform([selected_market])[0]
region_encoded = order_region_encoder.transform([selected_region])[0]

# Create input data dictionary
input_data = {col: 0 for col in expected_columns}

# Update with numeric values
input_data.update({
    "Order Item Profit Ratio": order_item_profit_ratio,
    "Product Price": product_price,
    "Order Item Discount Rate": order_item_discount_rate,
    "Market": market_encoded,
    "Order Region": region_encoded
})

# One-hot encoding for department
department_encoded = f"Department Name_{selected_department}"
if department_encoded in input_data:
    input_data[department_encoded] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure column order matches model training
input_df = input_df.reindex(columns=expected_columns, fill_value=0)

# Predict Button
if st.button("🚀 Predict Profit"):
    try:
        prediction = gb_model_final.predict(input_df)
        predicted_profit = prediction[0]

        # Classification based on product price
        if product_price > 30:
            classification = "🌟 High-Priority, High-Margin Order"
        else:
            classification = "💸 Lower-Margin Order"

        st.subheader("Predicted Profit (in USD):")
        st.markdown(f"### 💲 **${predicted_profit:.2f}**")
        st.success(f"✅ {classification}")
    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")
