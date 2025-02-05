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
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.1;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title("Profit Predictor: Enhancing Business Decisions with Data Science")

# Dataset Introduction
st.markdown("""
    ### About the Dataset
    The **DataCo Global Supply Chain Dataset** is a comprehensive resource designed to analyze and optimize key aspects of the **supply chain**. By leveraging **Machine Learning Algorithms** and **Data Science techniques**, this dataset provides actionable insights for businesses in areas such as:

    - **Inventory Management**: Optimizing stock levels to meet demand without overstocking.
    - **Demand Forecasting**: Predicting sales trends and adjusting production schedules accordingly.
    - **Profitability Analysis**: Identifying factors that influence profits per order and highlighting areas for improvement.
    - **Sales Optimization**: Analyzing sales data to understand customer behavior and refine pricing strategies.

    With this dataset, businesses can transform raw data into strategic decisions, driving better operational efficiency, customer satisfaction, and overall profitability. By combining **structured data** with **advanced analytical models**, it empowers organizations to make data-driven decisions that streamline operations and improve outcomes.
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
