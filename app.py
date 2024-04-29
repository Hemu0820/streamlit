import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load your trained model
model_path = 'gradient_boosting_regressor.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize the MinMaxScaler for 'latitude' and 'longitude'
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.array([40.489497, -74.251999]), np.array([40.915568 - 40.489497, -73.700165 + 74.251999])

# Initialize OneHotEncoder for 'neighbourhood_group' and 'room_type' based on the categories seen during training
encoder = OneHotEncoder()
encoder.categories_ = [array_of_neighbourhood_groups, array_of_room_types]  # replace with actual categories

# Creating a function to get user input
def user_input_features():
    latitude = st.number_input('Latitude', format="%.6f")
    longitude = st.number_input('Longitude', format="%.6f")
    minimum_nights = st.number_input('Minimum Nights', format="%d")
    number_of_reviews = st.number_input('Number of Reviews', format="%d")
    reviews_per_month = st.number_input('Reviews per Month', format="%.2f")
    calculated_host_listings_count = st.number_input('Calculated Host Listings Count', format="%d")
    availability_365 = st.number_input('Availability 365', format="%d")
    neighbourhood_group = st.selectbox('Neighbourhood Group', options=array_of_neighbourhood_groups)
    room_type = st.selectbox('Room Type', options=array_of_room_types)

    data = {
        'latitude': [latitude],
        'longitude': [longitude],
        'minimum_nights': [minimum_nights],
        'number_of_reviews': [number_of_reviews],
        'reviews_per_month': [reviews_per_month],
        'calculated_host_listings_count': [calculated_host_listings_count],
        'availability_365': [availability_365],
        'neighbourhood_group': [neighbourhood_group],
        'room_type': [room_type]
    }
    features = pd.DataFrame(data)
    
    # Apply Min-Max scaling to latitude and longitude
    features[['latitude', 'longitude']] = scaler.transform(features[['latitude', 'longitude']])
    
    # One-hot encode 'neighbourhood_group' and 'room_type'
    encoded_features = encoder.transform(features[['neighbourhood_group', 'room_type']]).toarray()
    features = features.drop(['neighbourhood_group', 'room_type'], axis=1)
    features = np.hstack([features, encoded_features])
    
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
