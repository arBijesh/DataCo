# %% 
import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('gb_model_final.pkl', 'rb') as model_file:
    gb_model_final = pickle.load(model_file)

# Streamlit UI
st.set_page_config(page_title="Profit Predictor", page_icon="💰", layout="centered")
st.title("Profit Predictor: Enhancing Business Decisions with Data Science")

# Introduction and explanation
st.markdown("""
    This app uses a machine learning model to predict the **Profit per Order** based on 
    key business features. Simply enter the details below, and the model will predict 
    the expected profit in dollars for a given set of inputs.
""")

# User inputs for the top 6 features with tooltips
order_item_profit_ratio = st.number_input(
    "Order Item Profit Ratio", min_value=0.0, value=0.1, step=0.01,
    help="Ratio of profit per order item relative to the cost."
)
sales_per_customer = st.number_input(
    "Sales per customer", min_value=0.0, value=100.0, step=1.0,
    help="Average sales per customer."
)
order_item_total = st.number_input(
    "Order Item Total", min_value=0.0, value=500.0, step=10.0,
    help="Total price of all items in the order."
)
order_item_product_price = st.number_input(
    "Order Item Product Price", min_value=0.0, value=50.0, step=1.0,
    help="Price of a single product in the order."
)
sales = st.number_input(
    "Sales", min_value=0.0, value=1000.0, step=10.0,
    help="Total sales amount generated."
)
product_price = st.number_input(
    "Product Price", min_value=0.0, value=200.0, step=1.0,
    help="Price of a single product."
)

# Add a section header for the prediction
st.markdown("### Enter the values and click **'Predict'** to get the profit prediction.")

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

# %% 
