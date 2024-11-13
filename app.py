import streamlit as st
import pickle
import numpy as np
import folium as fl
from streamlit_folium import st_folium
from folium import Popup

# Load LinearRegression model di awal sebelum digunakan
with open('LinearRegression.pkl', 'rb') as file:
    LinearRegression_Model = pickle.load(file)
    
# Load scaler
with open('StandardScalerUber.pkl', 'rb') as file:
    scaler = pickle.load(file)

def main():
    st.sidebar.title("Uber Fare Prediction")
    menu = ["Predict Fare", "Credit"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Predict Fare":
        run_prediction_app()
    elif choice == "Credit":
        st.subheader("Credit")
        st.write("This application was developed by me and my data science bootcamp team named tech-leph4nt. I added a few modification on this application page for better user experience. Thank you for trying this application!")
    

def run_prediction_app():
    # Select pickup location
    st.subheader("Select pickup location")

    # Show map on page
    m = fl.Map(
        tiles="OpenStreetMap",
        zoom_start=11,
        location=[-6.1762, 106.8274],
    )
    # Show popup location
    m.add_child(fl.LatLngPopup())
    map_ny = st_folium(m, height=400, width=700)

    # Save lat and long
    if map_ny["last_clicked"]:
        pickup_lat = map_ny["last_clicked"]["lat"]
        pickup_long = map_ny["last_clicked"]["lng"]

    # Select dropoff location
    st.subheader("Select dropoff location")

    # Show map on page
    m2 = fl.Map(
        tiles="OpenStreetMap",
        zoom_start=11,
        location=[-6.3762, 106.8274],
    )
    # Show popup location
    m2.add_child(fl.LatLngPopup())
    map_ny2 = st_folium(m2, height=400, width=700)

    # Save lat and long
    if map_ny2["last_clicked"]:
        dropoff_lat = map_ny2["last_clicked"]["lat"]
        dropoff_long = map_ny2["last_clicked"]["lng"]

    # Calculate distance using Haversine formula
    if (map_ny["last_clicked"]) and (map_ny2["last_clicked"]):
        distance = haversine_array(pickup_long, pickup_lat, dropoff_long, dropoff_lat)
        st.write(f"Calculated Distance: {distance:.2f} km")

    # Predict fare using the best model
    if st.button('Predict Fare'):
        try:
            distance = scaler.transform([[distance]])
            fare_pred = LinearRegression_Model.predict(distance)
            st.write(f"Predicted Fare: ${fare_pred[0]:.2f}")
        except:
            st.write("Please select pickup and dropoff location first.")

def haversine_array(pickup_long, pickup_lat, dropoff_long, dropoff_lat):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    pickup_long, pickup_lat, dropoff_long, dropoff_lat = map(lambda x: x/360.*(2*np.pi), [pickup_long, pickup_lat, dropoff_long, dropoff_lat])
    # haversine formula
    dlon = dropoff_long - pickup_long
    dlat = dropoff_lat - pickup_lat
    a = np.sin(dlat/2)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

if __name__ == "__main__":
    main()
