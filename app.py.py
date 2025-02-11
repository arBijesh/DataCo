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
order_country_options = ["Indonesia", "India", "Australia", "China", "Japón",
                         "Corea del Sur", "Singapur", "Turquía", "Mongolia",
                         "Estados Unidos", "Nigeria", "República Democrática del Congo",
                         "Senegal", "Marruecos", "Alemania", "Francia", "Países Bajos",
                         "Reino Unido", "Guatemala", "El Salvador", "Panamá",
                         "República Dominicana", "Venezuela", "Colombia", "Honduras",
                         "Brasil", "México", "Uruguay", "Argentina", "Cuba", "Perú",
                         "Nicaragua", "Ecuador", "Angola", "Sudán", "Somalia",
                         "Costa de Marfil", "Egipto", "Italia", "España", "Suecia",
                         "Austria", "Canada", "Madagascar", "Argelia", "Liberia", "Zambia",
                         "Níger", "SudAfrica", "Mozambique", "Tanzania", "Ruanda", "Israel",
                         "Nueva Zelanda", "Bangladés", "Tailandia", "Irak", "Arabia Saudí",
                         "Filipinas", "Kazajistán", "Irán", "Myanmar (Birmania)",
                         "Uzbekistán", "Benín", "Camerún", "Kenia", "Togo", "Ucrania",
                         "Polonia", "Portugal", "Rumania", "Trinidad y Tobago",
                         "Afganistán", "Pakistán", "Vietnam", "Malasia", "Finlandia",
                         "Rusia", "Irlanda", "Noruega", "Eslovaquia", "Bélgica", "Bolivia",
                         "Chile", "Jamaica", "Yemen", "Ghana", "Guinea", "Etiopía",
                         "Bulgaria", "Kirguistán", "Georgia", "Nepal",
                         "Emiratos Árabes Unidos", "Camboya", "Uganda", "Lesoto",
                         "Lituania", "Suiza", "Hungría", "Dinamarca", "Haití",
                         "Bielorrusia", "Croacia", "Laos", "Baréin", "Macedonia",
                         "República Checa", "Sri Lanka", "Zimbabue", "Eritrea",
                         "Burkina Faso", "Costa Rica", "Libia", "Barbados", "Tayikistán",
                         "Siria", "Guadalupe", "Papúa Nueva Guinea", "Azerbaiyán",
                         "Turkmenistán", "Paraguay", "Jordania", "Hong Kong", "Martinica",
                         "Moldavia", "Qatar", "Mali", "Albania", "República del Congo",
                         "Bosnia y Herzegovina", "Omán", "Túnez", "Sierra Leona", "Yibuti",
                         "Burundi", "Montenegro", "Gabón", "Sudán del Sur", "Luxemburgo",
                         "Namibia", "Mauritania", "Grecia", "Suazilandia", "Guyana",
                         "Guayana Francesa", "República Centroafricana", "Taiwán",
                         "Estonia", "Líbano", "Chipre", "Guinea-Bissau", "Surinam",
                         "Belice", "Eslovenia", "República de Gambia", "Botsuana",
                         "Armenia", "Guinea Ecuatorial", "Kuwait", "Bután", "Chad",
                         "Serbia", "Sáhara Occidental"]

# UI Inputs
selected_market = st.selectbox("🌎 Market", market_options)
selected_region = st.selectbox("📍 Order Region", order_region_options)
selected_country = st.selectbox("🏳️ Order Country", order_country_options)
selected_department = st.selectbox("🏪 Department Name", department_options)

# Encode categorical variables
encoded_market = market_encoder.transform([selected_market])[0]
encoded_region = order_region_encoder.transform([selected_region])[0]
encoded_country = order_country_encoder.transform([selected_country])[0]

# Prepare input data
input_data = np.array([...]).reshape(1, -1)

# Predict Button
if st.button("🚀 Predict Profit"):
    prediction = model.predict(input_data)
    profit_label = "High Profit Order" if prediction[0] > 50 else "Low Profit Order"
    st.subheader("Predicted Profit (in USD):")
    st.markdown(f"### 💲 **${prediction[0]:.2f}** - {profit_label}")
    st.success("✅ Prediction Successful!")
