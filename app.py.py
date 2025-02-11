import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set up the Streamlit page
st.set_page_config(page_title="Profit Predictor - DataCo Supply Chain", page_icon="📦", layout="wide")

st.title("🚀 Profit Predictor: Enhancing Business Decisions with Data Science")
st.markdown("### 📊 Enter Order Details to Predict Profit:")

# Load trained model
with open("gb_model_final_hyperparameter.pkl", "rb") as model_file:
    gb_model_final = pickle.load(model_file)

# Load encoders
market_encoder = pickle.load(open("market_encoder.pkl", "rb"))
order_region_encoder = pickle.load(open("order_region_encoder.pkl", "rb"))
order_country_encoder = pickle.load(open("order_country_encoder.pkl", "rb"))

# Get expected feature names from the trained model
expected_columns = list(gb_model_final.feature_names_in_)

# Define categorical options
market_options = ["Europe", "LATAM", "Pacific Asia", "USCA"]
department_options = ["Book Shop", "Discs Shop", "Fan Shop", "Fitness", "Footwear",
                      "Golf", "Health and Beauty", "Outdoors", "Pet Shop", "Technology"]
order_country_options = ['Indonesia', 'India', 'Australia', 'China', 'Japón',
       'Corea del Sur', 'Singapur', 'Turquía', 'Mongolia',
       'Estados Unidos', 'Nigeria', 'República Democrática del Congo',
       'Senegal', 'Marruecos', 'Alemania', 'Francia', 'Países Bajos',
       'Reino Unido', 'Guatemala', 'El Salvador', 'Panamá',
       'República Dominicana', 'Venezuela', 'Colombia', 'Honduras',
       'Brasil', 'México', 'Uruguay', 'Argentina', 'Cuba', 'Perú',
       'Nicaragua', 'Ecuador', 'Angola', 'Sudán', 'Somalia',
       'Costa de Marfil', 'Egipto', 'Italia', 'España', 'Suecia',
       'Austria', 'Canada', 'Madagascar', 'Argelia', 'Liberia', 'Zambia',
       'Níger', 'SudAfrica', 'Mozambique', 'Tanzania', 'Ruanda', 'Israel',
       'Nueva Zelanda', 'Bangladés', 'Tailandia', 'Irak', 'Arabia Saudí',
       'Filipinas', 'Kazajistán', 'Irán', 'Myanmar (Birmania)',
       'Uzbekistán', 'Benín', 'Camerún', 'Kenia', 'Togo', 'Ucrania',
       'Polonia', 'Portugal', 'Rumania', 'Trinidad y Tobago',
       'Afganistán', 'Pakistán', 'Vietnam', 'Malasia', 'Finlandia',
       'Rusia', 'Irlanda', 'Noruega', 'Eslovaquia', 'Bélgica', 'Bolivia',
       'Chile', 'Jamaica', 'Yemen', 'Ghana', 'Guinea', 'Etiopía',
       'Bulgaria', 'Kirguistán', 'Georgia', 'Nepal',
       'Emiratos Árabes Unidos', 'Camboya', 'Uganda', 'Lesoto',
       'Lituania', 'Suiza', 'Hungría', 'Dinamarca', 'Haití',
       'Bielorrusia', 'Croacia', 'Laos', 'Baréin', 'Macedonia',
       'República Checa', 'Sri Lanka', 'Zimbabue', 'Eritrea',
       'Burkina Faso', 'Costa Rica', 'Libia', 'Barbados', 'Tayikistán',
       'Siria', 'Guadalupe', 'Papúa Nueva Guinea', 'Azerbaiyán',
       'Turkmenistán', 'Paraguay', 'Jordania', 'Hong Kong', 'Martinica',
       'Moldavia', 'Qatar', 'Mali', 'Albania', 'República del Congo',
       'Bosnia y Herzegovina', 'Omán', 'Túnez', 'Sierra Leona', 'Yibuti',
       'Burundi', 'Montenegro', 'Gabón', 'Sudán del Sur', 'Luxemburgo',
       'Namibia', 'Mauritania', 'Grecia', 'Suazilandia', 'Guyana',
       'Guayana Francesa', 'República Centroafricana', 'Taiwán',
       'Estonia', 'Líbano', 'Chipre', 'Guinea-Bissau', 'Surinam',
       'Belice', 'Eslovenia', 'República de Gambia', 'Botsuana',
       'Armenia', 'Guinea Ecuatorial', 'Kuwait', 'Bután', 'Chad',
       'Serbia', 'Sáhara Occidental']  # Add real country options

# UI Inputs
selected_market = st.selectbox("🌎 Market", market_options)
selected_country = st.selectbox("🌍 Order Country", order_country_options)
selected_department = st.selectbox("🏪 Department Name", department_options)

order_item_profit_ratio = st.number_input("💰 Order Item Profit Ratio", min_value=0.0, value=0.1, step=0.01)
order_item_total = st.number_input("📦 Order Item Total", min_value=0.0, value=500.0, step=10.0)
sales_per_customer = st.number_input("👤 Sales per customer", min_value=0.0, value=100.0, step=1.0)
order_item_product_price = st.number_input("🏷 Order Item Product Price", min_value=0.0, value=50.0, step=1.0)
sales = st.number_input("📈 Sales", min_value=0.0, value=1000.0, step=10.0)
product_price = st.number_input("💲 Product Price", min_value=0.0, value=200.0, step=1.0)

# Create input data dictionary
input_data = {col: 0 for col in expected_columns}

# Update with numeric values
input_data.update({
    "Order Item Profit Ratio": order_item_profit_ratio,
    "Order Item Total": order_item_total,
    "Sales per customer": sales_per_customer,
    "Order Item Product Price": order_item_product_price,
    "Sales": sales,
    "Product Price": product_price
})

# Encode categorical features (One-Hot Encoding)
market_encoded = f"Market_{selected_market}"
department_encoded = f"Department Name_{selected_department}"
country_encoded = f"Order Country_{selected_country}"  # New country feature

if market_encoded in input_data:
    input_data[market_encoded] = 1
if department_encoded in input_data:
    input_data[department_encoded] = 1
if country_encoded in input_data:  # Handle Order Country encoding
    input_data[country_encoded] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure column order matches model training
input_df = input_df.reindex(columns=expected_columns, fill_value=0)

# Debugging: Show expected vs. actual features
st.write("Expected Features:", expected_columns)
st.write("Provided Features:", input_df.columns.tolist())

# Predict Button
if st.button("🚀 Predict Profit"):
    try:
        prediction = gb_model_final.predict(input_df)
        st.subheader("Predicted Profit (in USD):")
        st.markdown(f"### 💲 **${prediction[0]:.2f}**")
        st.success("✅ Prediction Successful!")
    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")
