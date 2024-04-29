import streamlit as st
import pandas as pd
import pickle

# Assuming the model is loaded from a pickle file
model_path = 'gradient_boosting_regressor.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

def user_input_features():
    # Default values
    latitude_default = 40.7128  # Example value, replace with your default
    longitude_default = -74.0060  # Example value, replace with your default
    neighbourhood_group_default = 'Manhattan'  # Example value, replace with your default
    room_type_default = 'Entire home/apt'  # Example value, replace with your default
    
    # User inputs for other features
    minimum_nights = st.number_input('Minimum Nights', format="%d")
    number_of_reviews = st.number_input('Number of Reviews', format="%d")
    reviews_per_month = st.number_input('Reviews per Month', format="%.2f")
    calculated_host_listings_count = st.number_input('Calculated Host Listings Count', format="%d")
    availability_365 = st.number_input('Availability 365', format="%d")
    
    # Create data dictionary including hidden default values
    data = {
        'latitude': latitude_default,
        'longitude': longitude_default,
        'minimum_nights': minimum_nights,
        'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month,
        'calculated_host_listings_count': calculated_host_listings_count,
        'availability_365': availability_365,
        'neighbourhood_group': neighbourhood_group_default,
        'room_type': room_type_default
    }
    
    # Assuming one-hot encoding for categorical variables
    data.update({
        'neighbourhood_group': 1 if neighbourhood_group_default == 'Manhattan' else 0,
        'neighbourhood_group': 2 if neighbourhood_group_default == 'Brooklyn' else 0,
        'neighbourhood_group': 3 if neighbourhood_group_default == 'Queens' else 0,
        'neighbourhood_group': 4 if neighbourhood_group_default == 'Staten Island' else 0,
        'neighbourhood_group': 5 if neighbourhood_group_default == 'Bronx' else 0,
        'room_type': 1 if room_type_default == 'Entire home/apt' else 0,
        'room_type': 2 if room_type_default == 'Private room' else 0,
        'room_type': 3 if room_type_default == 'Shared room' else 0
    })

    features = pd.DataFrame(data, index=[0])
    return features

def main():
    st.write("# Price Prediction App")
    st.write("This app predicts prices based on input features using a model.")
    
    df = user_input_features()

    if st.button('Predict'):
        prediction = model.predict(df)
        st.write(f'### Estimated Price: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    main()
