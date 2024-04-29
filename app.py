import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load your trained model
model_path = 'gradient_boosting_regressor.pkl"'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize the MinMaxScaler for 'latitude' and 'longitude'
# Replace 'min_lat', 'max_lat', 'min_long', and 'max_long' with the actual values used during training
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.array([40.489497, -74.251999]), np.array([40.915568 - 40.489497, -73.700165 + 74.251999])

# Creating a function to get user input
def user_input_features():
    latitude = st.number_input('Latitude', format="%.6f")
    longitude = st.number_input('Longitude', format="%.6f")
    minimum_nights = st.number_input('Minimum Nights', format="%d")
    number_of_reviews = st.number_input('Number of Reviews', format="%d")
    reviews_per_month = st.number_input('Reviews per Month', format="%.2f")
    calculated_host_listings_count = st.number_input('Calculated Host Listings Count', format="%d")
    availability_365 = st.number_input('Availability 365', format="%d")

    data = {
        'latitude': latitude,
        'longitude': longitude,
        'minimum_nights': minimum_nights,
        'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month,
        'calculated_host_listings_count': calculated_host_listings_count,
        'availability_365': availability_365
    }
    features = pd.DataFrame(data, index=[0])
    # Apply Min-Max scaling to latitude and longitude
    features[['latitude', 'longitude']] = scaler.transform(features[['latitude', 'longitude']])
    return features

# Streamlit main function
def main():
    st.write("# Price Prediction App")
    st.write("This app predicts prices based on input features using a Gradient Boosting Regressor model.")
    
    df = user_input_features()

    if st.button('Predict'):
        prediction = model.predict(df)
        st.write(f'### Estimated Price: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    main()

