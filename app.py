import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler  # Assuming StandardScaler was used

# Load your trained model and scaler
model_path = 'gradient_boosting_regressor.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

scaler_path = 'scaler.pkl'  # Assuming the scaler is saved separately
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

def user_input_features():
    # User inputs (hidden and visible)
    latitude = 40.7128  # default value, hidden
    longitude = -74.0060  # default value, hidden

    # Dropdowns for neighbourhood_group and room_type
    neighbourhood_group = st.selectbox(
        'Neighbourhood Group',
        ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx'])
    room_type = st.selectbox(
        'Room Type',
        ['Private room', 'Entire home/apt', 'Shared room'])

    # Mapping the inputs to numeric values
    neighbourhood_group_mapping = {
        'Brooklyn': 1,
        'Manhattan': 2,
        'Queens': 3,
        'Staten Island': 4,
        'Bronx': 5
    }
    room_type_mapping = {
        'Private room': 1,
        'Entire home/apt': 2,
        'Shared room': 3
    }

    neighbourhood_group = neighbourhood_group_mapping[neighbourhood_group]
    room_type = room_type_mapping[room_type]

    minimum_nights = st.number_input('Minimum Nights', min_value=1, step=1)
    number_of_reviews = st.number_input('Number of Reviews', min_value=0, step=1)
    reviews_per_month = st.number_input('Reviews per Month', min_value=0.0, step=0.1)
    calculated_host_listings_count = st.number_input('Calculated Host Listings Count', min_value=0, step=1)
    availability_365 = st.number_input('Availability 365', min_value=0, max_value=365, step=1)

    # Create DataFrame
    data = {
        'latitude': [latitude],
        'longitude': [longitude],
        'neighbourhood_group': [neighbourhood_group],
        'room_type': [room_type],
        'minimum_nights': [minimum_nights],
        'number_of_reviews': [number_of_reviews],
        'reviews_per_month': [reviews_per_month],
        'calculated_host_listings_count': [calculated_host_listings_count],
        'availability_365': [availability_365]
    }
    features = pd.DataFrame(data)
    
    # Scale features
    scaled_features = scaler.transform(features)
    features = pd.DataFrame(scaled_features, columns=features.columns)
    
    return features

def main():
    st.write("# Price Prediction App")
    st.write("This app predicts prices based on input features using a machine learning model.")
    
    df = user_input_features()

    if st.button('Predict'):
        prediction = model.predict(df)
        st.write(f'### Estimated Price: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    main()
