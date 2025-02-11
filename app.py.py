import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set up the page
st.set_page_config(page_title="Profit Predictor - DataCo Supply Chain", page_icon="📦", layout="wide")

st.title("🚀 Profit Predictor: Enhancing Business Decisions with Data Science")
st.markdown("### 📊 Enter Order Details to Predict Profit:")

# Load trained Gradient Boosting model
with open("gb_model_final_hyperparameter.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load LabelEncoders
market_encoder = pickle.load(open("market_encoder.pkl", "rb"))
order_region_encoder = pickle.load(open("order_region_encoder.pkl", "rb"))
order_country_encoder = pickle.load(open("order_country_encoder.pkl", "rb"))

# Define categorical options
market_options = ["Europe", "LATAM", "Pacific Asia", "USCA"]
department_options = ["Book Shop", "Discs Shop", "Fan Shop", "Fitness", "Footwear",
                      "Golf", "Health and Beauty", "Outdoors", "Pet Shop", "Technology"]
order_region_options = ["Southeast Asia", "South Asia", "Oceania", "Eastern Asia",
                        "West Asia", "West of USA", "US Center", "West Africa",
                        "Central Africa", "North Africa", "Western Europe",
                        "Northern Europe", "Central America", "Caribbean", "South America",
                        "East Africa", "Southern Europe", "East of USA", "Canada",
                        "Southern Africa", "Central Asia", "Eastern Europe",
                        "South of USA"]
order_country_options = ["Indonesia", "India", "Australia", "China", "Japan",
                         "South Korea", "Singapore", "Turkey", "Mongolia",
                         "United States", "Nigeria", "Democratic Republic of the Congo",
                         "Senegal", "Morocco", "Germany", "France", "Netherlands",
                         "United Kingdom", "Guatemala", "El Salvador", "Panama",
                         "Dominican Republic", "Venezuela", "Colombia", "Honduras",
                         "Brazil", "Mexico", "Uruguay", "Argentina", "Cuba", "Peru",
                         "Nicaragua", "Ecuador", "Angola", "Sudan", "Somalia",
                         "Ivory Coast", "Egypt", "Italy", "Spain", "Sweden",
                         "Austria", "Canada", "Madagascar", "Algeria", "Liberia", "Zambia",
                         "Niger", "South Africa", "Mozambique", "Tanzania", "Rwanda", "Israel",
                         "New Zealand", "Bangladesh", "Thailand", "Iraq", "Saudi Arabia",
                         "Philippines", "Kazakhstan", "Iran", "Myanmar",
                         "Uzbekistan", "Benin", "Cameroon", "Kenya", "Togo", "Ukraine",
                         "Poland", "Portugal", "Romania", "Trinidad and Tobago",
                         "Afghanistan", "Pakistan", "Vietnam", "Malaysia", "Finland",
                         "Russia", "Ireland", "Norway", "Slovakia", "Belgium", "Bolivia",
                         "Chile", "Jamaica", "Yemen", "Ghana", "Guinea", "Ethiopia",
                         "Bulgaria", "Kyrgyzstan", "Georgia", "Nepal",
                         "United Arab Emirates", "Cambodia", "Uganda", "Lesotho",
                         "Lithuania", "Switzerland", "Hungary", "Denmark", "Haiti",
                         "Belarus", "Croatia", "Laos", "Bahrain", "Macedonia",
                         "Czech Republic", "Sri Lanka", "Zimbabwe", "Eritrea",
                         "Burkina Faso", "Costa Rica", "Libya", "Barbados", "Tajikistan",
                         "Syria", "Guadeloupe", "Papua New Guinea", "Azerbaijan",
                         "Turkmenistan", "Paraguay", "Jordan", "Hong Kong", "Martinique",
                         "Moldova", "Qatar", "Mali", "Albania", "Republic of the Congo",
                         "Bosnia and Herzegovina", "Oman", "Tunisia", "Sierra Leone", "Djibouti",
                         "Burundi", "Montenegro", "Gabon", "South Sudan", "Luxembourg",
                         "Namibia", "Mauritania", "Greece", "Eswatini", "Guyana",
                         "French Guiana", "Central African Republic", "Taiwan",
                         "Estonia", "Lebanon", "Cyprus", "Guinea-Bissau", "Suriname",
                         "Belize", "Slovenia", "Gambia", "Botswana",
                         "Armenia", "Equatorial Guinea", "Kuwait", "Bhutan", "Chad",
                         "Serbia", "Western Sahara"]

# UI Inputs
selected_market = st.selectbox("🌎 Market", market_options)
selected_region = st.selectbox("📍 Order Region", order_region_options)
selected_country = st.selectbox("🏳️ Order Country", order_country_options)
selected_department = st.selectbox("🏪 Department Name", department_options)

# Set default profit ratio based on department
profit_ratio_defaults = {
    "Book Shop": 0.3,
    "Discs Shop": 0.25,
    "Fan Shop": 0.35,
    "Fitness": 0.4,
    "Footwear": 0.45,
    "Golf": 0.5,
    "Health and Beauty": 0.55,
    "Outdoors": 0.6,
    "Pet Shop": 0.3,
    "Technology": 0.65
}

profit_ratio = st.slider("📈 Profit Ratio", min_value=0.0, max_value=1.0, step=0.01, value=profit_ratio_defaults.get(selected_department, 0.3))
product_price = st.number_input("💰 Product Price", min_value=0.0, step=0.01)
discount_rate = st.slider("🎯 Order Item Discount Rate", min_value=0.0, max_value=1.0, step=0.01)

# Encode categorical variables
encoded_market = market_encoder.transform([selected_market])[0]
encoded_region = order_region_encoder.transform([selected_region])[0]
encoded_country = order_country_encoder.transform([selected_country])[0]

# Prepare department encoding
department_encoding = [1 if dept == selected_department else 0 for dept in department_options]

# Define feature names (must match training data feature names)
feature_names = ["Market", "Order_Region", "Order_Country", "Profit_Ratio", "Product_Price", "Discount_Rate"] + department_options

# Convert input data into a DataFrame
input_data = pd.DataFrame([[
    encoded_market, encoded_region, encoded_country, 
    profit_ratio, product_price, discount_rate
] + department_encoding], columns=feature_names)

# Predict Button
if st.button("🚀 Predict Profit"):
    prediction = model.predict(input_data)
    profit_label = "High Profit Order" if prediction[0] > 50 else "Low Profit Order"
    st.subheader("Predicted Profit (in USD):")
    st.markdown(f"### 💲 **${prediction[0]:.2f}** - {profit_label}")
    st.success("✅ Prediction Successful!")
