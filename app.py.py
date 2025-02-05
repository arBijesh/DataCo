import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('gb_model_final.pkl', 'rb') as model_file:
    gb_model_final = pickle.load(model_file)

# Streamlit UI setup
st.set_page_config(page_title="Profit Predictor - DataCo Supply Chain", page_icon="📦", layout="wide")

# Add custom CSS for watermark effect
st.markdown("""
    <style>
        .watermark {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            opacity: 0.1;
            z-index: -1;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title("Profit Predictor: Enhancing Business Decisions with Data Science")

# Dataset Introduction
st.markdown("""
    ### About the DataSet
    The dataset used in this analysis is provided by **DataCo Global** and pertains to the **Supply Chain** domain. 
    It allows the use of **Machine Learning Algorithms** and **R Software** for analysis, offering insights into key areas such as:
    
    - **Provisioning**
    - **Production**
    - **Sales**
    - **Commercial Distribution**

    Additionally, the dataset allows the correlation of **Structured Data** with **Unstructured Data** for knowledge generation, making it a valuable resource for Supply Chain analysis.
""")

# Add an image as watermark in the background
st.markdown('<img src="supply_chain_image.jpg" class="watermark">', unsafe_allow_html=True)

# User Inputs and Prediction Section
st.markdown("### Enter the values below to predict the profit per order:")

# User inputs for the top 6 features
order_item_profit_ratio = st.number_input("Order Item Profit Ratio", min_value=0.0, value=0.1, step=0.01)
sales_per_customer = st.number_input("Sales per customer", min_value=0.0, value=100.0, step=1.0)
order_item_total = st.number_input("Order Item Total", min_value=0.0, value=500.0, step=10.0)
order_item_product_price = st.number_input("Order Item Product Price", min_value=0.0, value=50.0, step=1.0)
sales = st.number_input("Sales", min_value=0.0, value=1000.0, step=10.0)
product_price = st.number_input("Product Price", min_value=0.0, value=200.0, step=1.0)

# Make prediction when the button is pressed
if st.button("Predict"):
    # Create input data as a 2D array for prediction
    input_data = np.array([[order_item_profit_ratio, sales_per_customer, order_item_total,
                            order_item_product_price, sales, product_price]])
    
    # Predict using the trained model
    prediction = gb_model_final.predict(input_data)
    
    # Display the prediction with a formatted dollar amount
    st.subheader("Predicted Profit per Order (in USD):")
    st.markdown(f"### 💲 **${prediction[0]:.2f}**")

    # Add a success message for better user experience
    st.success("The model has successfully made a prediction!")
