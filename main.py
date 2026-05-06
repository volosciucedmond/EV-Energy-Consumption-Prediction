import streamlit as st
import pandas as pd
import joblib
import sys
import os

# setup paths to include the source code
sys.path.append(os.path.abspath('src'))
from features import create_physics_features, calculate_carbon_footprint

# load the brain
@st.cache_resource

def load_assets():
    model = joblib.load('models/ev_model.pkl')
    model_columns = joblib.load('models/model_columns.pkl')
    return model, model_columns

model, model_columns = load_assets()

# ui layout
st.title("EV Energy Simulator")
st.markdown("Predict consumption and optimize efficiency in real-time.")

# sidebar for user inputs
st.sidebar.header("Driving Conditions")
speed = st.sidebar.slider("Speed (km/h)", 0, 130, 60)
slope = st.sidebar.slider("Road Incline (%)", -5.0, 10.0, 0.0)
temp = st.sidebar.slider("Battery Temp (°C)", 10, 45, 25)
weight = st.sidebar.number_input("Vehicle Weight (kg)", 1200, 2500, 1800)

# categorical inputs 
st.sidebar.header("Vehicle State")
mode = st.sidebar.selectbox("Driving Mode", [1, 2, 3], format_func=lambda x: ["Eco", "Normal", "Sport"][x-1])
traffic = st.sidebar.selectbox("Traffic", [1, 2, 3], format_func=lambda x: ["Low", "Medium", "High"][x-1])

# simulation logic
if st.button("Run Simulation"):
    # create a raw input dataframe
    input_df = pd.DataFrame([{
        'Speed_kmh': speed,
        'Slope_%': slope,
        'Battery_Temperature_C': temp,
        'Vehicle_Weight_kg': weight,
        'Driving_Mode': mode,
        'Traffic_Condition': traffic,
        'Acceleration_ms2': 0, # assume steady state
        'Battery_State_%': 50,
        'Tire_Pressure_psi': 32,
        'Weather_Condition': 1,
        'Road_Type': 1,
        'Distance_Travelled_km': 10 # predict for a 10km stretch
    }])

    # apply pshyics logic
    input_df = create_physics_features(input_df)

    # handle One-Hot Encoding 
    # this creates a dataframe of zeros with the exact columns the model expects
    encoded_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # fill in the numerical values
    for col in input_df.columns:
        if col in model_columns:
            encoded_df[col] = input_df[col].values[0]
            
    # manually set the One-Hot dummies for mode and traffic
    if f"Driving_Mode_{mode}" in model_columns: encoded_df[f"Driving_Mode_{mode}"] = 1
    if f"Traffic_Condition_{traffic}" in model_columns: encoded_df[f"Traffic_Condition_{traffic}"] = 1

    # predict
    raw_prediction = model.predict(encoded_df)[0]
    
    # real world calibration
    # since the dataset is ~4x higher than a standard consumer EV, 
    # I apply a 0.25 calibration factor to bring it into "passenger car" territory
    calibration_factor = 0.25 
    prediction = raw_prediction * calibration_factor
    
    # display results
    col1, col2, col3 = st.columns(3)
    
    # calculate efficiency per 100km based on our 10km simulation
    efficiency_100km = (prediction / 10) * 100 
    
    with col1:
        st.metric("Trip Energy (10km)", f"{prediction:.2f} kWh")
    
    with col2:
        st.metric("Efficiency", f"{efficiency_100km:.1f} kWh/100km")
        
    with col3:
        co2 = prediction * 0.4 
        st.metric("Carbon Footprint", f"{co2:.2f} kg CO2")
        
        # dynamic drive advice
    st.subheader("💡 Personalised Driving Tips")
    
    # define an advice list to show multiple tips if they apply
    tips = []

    # aerodynamic Tip (based on speed and drag proxy)
    if speed > 100:
        tips.append("🚩 **High Speed Drag:** At speeds over 100 km/h, air resistance is your biggest enemy. Dropping to 90 km/h could extend your range by up to 15%.")
    elif speed < 30 and speed > 0:
        tips.append("ℹ️ **Low Speed Overhead:** You are moving slowly. While drag is low, auxiliary systems (AC/Lights) are a larger percentage of your total energy use.")

    # incline tip (based on slope)
    if slope > 3:
        tips.append("⛰️ **Climb Management:** Maintain a steady throttle. Avoid aggressive overtaking while climbing to prevent high-current battery heat.")
    elif slope < -2:
        tips.append("🔋 **Regen Opportunity:** On this downhill stretch, ensure you use 'B-Mode' or regenerative braking to put energy back into the pack.")

    # thermal/climate tip (based on battery temp)
    if temp < 15:
        tips.append("❄️ **Cold Weather:** Battery efficiency drops in the cold. Consider pre-heating the cabin while the car is still plugged in to save range.")
    elif temp > 35:
        tips.append("🔥 **High Heat:** The cooling system is working hard. Try to park in the shade to help the Battery Management System (BMS) stay efficient.")

    # mode tip
    if mode == 3: # sport mode
        tips.append("🏎️ **Sport Mode Active:** This provides peak torque but reduces efficiency. Switch to Eco for maximum range.")

    # display tips
    if not tips:
        st.success("✅ Your current settings are optimal for balanced efficiency.")
    else:
        for t in tips:
            st.info(t)