# %%
import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('gb_model_final.pkl', 'rb') as model_file:
    gb_model_final = pickle.load(model_file)

# Streamlit UI
st.title("Predicting with Final Gradient Boosting Model")

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
    
    # Display the prediction
    st.subheader("Predicted Target Value:")
    st.write(f"Predicted Value: {prediction[0]:.2f}")
# %%