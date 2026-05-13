import streamlit as st
import pandas as pd
import joblib
import sys
import os
import numpy as np

# setup and assets
sys.path.append(os.path.abspath('src'))
from features import create_physics_features

# load the model and columns
@st.cache_resource
def load_assets():
    model = joblib.load('models/ev_model.pkl')
    model_columns = joblib.load('models/model_columns.pkl')
    return model, model_columns

model, model_columns = load_assets()

# helper functions
def plot_efficiency_curve(model, base_input_df, model_columns, calibration_factor):
    """Generates data for the speed vs efficiency graph"""
    speeds = np.linspace(10, 150, 25)
    results = []
    
    for s in speeds:
        # create a fresh copy of the raw inputs
        temp_df = base_input_df.copy()
        temp_df['Speed_kmh'] = s
        
        # re-run physics engineering so Drag_Proxy and Incline_Work update for this speed
        temp_df = create_physics_features(temp_df)
        
        # encode for model
        encoded_temp = pd.DataFrame(0, index=[0], columns=model_columns)
        for col in temp_df.columns:
            if col in model_columns:
                encoded_temp[col] = temp_df[col].values[0]
        
        # ensure categorical dummies are set
        mode_val = base_input_df['Driving_Mode'].values[0]
        traffic_val = base_input_df['Traffic_Condition'].values[0]
        if f"Driving_Mode_{mode_val}" in model_columns: encoded_temp[f"Driving_Mode_{mode_val}"] = 1
        if f"Traffic_Condition_{traffic_val}" in model_columns: encoded_temp[f"Traffic_Condition_{traffic_val}"] = 1

        # predict
        pred = model.predict(encoded_temp)[0] * calibration_factor
        # efficiency (kWh/100km) = (kWh / 10km) * 10
        eff = (pred / 10) * 100 
        results.append(eff)
    
    chart_data = pd.DataFrame({
        'Speed (km/h)': speeds,
        'Efficiency (kWh/100km)': results
    }).set_index('Speed (km/h)')
    
    return chart_data

# ui layout
st.set_page_config(page_title="EV Digital Twin", layout="wide")
st.title("🚗 EV Energy Digital Twin Simulator")
st.markdown("Predict consumption and optimise efficiency using physics-informed AI.")

# sidebar for user inputs
st.sidebar.header("Driving Conditions")
speed = st.sidebar.slider("Speed (km/h)", 0, 150, 60)
slope = st.sidebar.slider("Road Incline (%)", -5.0, 10.0, 0.0)
temp = st.sidebar.slider("Battery Temp (°C)", 10, 45, 25)
weight = st.sidebar.number_input("Vehicle Weight (kg)", 1200, 2500, 1800)

st.sidebar.header("🔋 Vehicle State")
vehicle_type = st.sidebar.selectbox(
    "Select vehicle profile",
    ["Standard passenger EV (Calibrated)", "High-Load / Prototype (Raw Data)"]
)
mode = st.sidebar.selectbox("Driving Mode", [1, 2, 3], format_func=lambda x: ["Eco", "Normal", "Sport"][x-1])
traffic = st.sidebar.selectbox("Traffic", [1, 2, 3], format_func=lambda x: ["Low", "Medium", "High"][x-1])

# simulation logic
if st.sidebar.button("Run simulation"):
    # create raw input dataframe
    raw_input_df = pd.DataFrame([{
        'Speed_kmh': speed,
        'Slope_%': slope,
        'Battery_Temperature_C': temp,
        'Vehicle_Weight_kg': weight,
        'Driving_Mode': mode,
        'Traffic_Condition': traffic,
        'Acceleration_ms2': 0,
        'Battery_State_%': 50,
        'Tire_Pressure_psi': 32,
        'Weather_Condition': 1,
        'Road_Type': 1,
        'Distance_Travelled_km': 10 
    }])

    # step 1: apply physics engineering
    engineered_df = create_physics_features(raw_input_df.copy())

    # step 2: one-hot encoding alignment
    encoded_df = pd.DataFrame(0, index=[0], columns=model_columns)
    for col in engineered_df.columns:
        if col in model_columns:
            encoded_df[col] = engineered_df[col].values[0]
    
    if f"Driving_Mode_{mode}" in model_columns: encoded_df[f"Driving_Mode_{mode}"] = 1
    if f"Traffic_Condition_{traffic}" in model_columns: encoded_df[f"Traffic_Condition_{traffic}"] = 1

    # step 3: prediction & calibration
    raw_prediction = model.predict(encoded_df)[0]
    calibration_factor = 0.25 if "Standard" in vehicle_type else 1.0
    prediction = raw_prediction * calibration_factor
    
    # efficiency & savings calculations
    efficiency_100km = (prediction / 10) * 100
    ice_co2 = 1.84 # average petrol car (8L/100km) CO2 for 10km
    ev_co2 = prediction * 0.4
    co2_savings = ice_co2 - ev_co2

    # result display
    st.divider()
    
    if speed == 0:
        st.info("📢 **Stationary Vehicle:** The car is powered on but not moving. Energy is being consumed primarily by auxiliary systems (HVAC, Electronics).")
    
    # performance metrics (top row)
    st.subheader("📊 Performance Metrics")
    m1, m2, m3 = st.columns(3)
    
    m1.metric("Trip Energy (10km)", f"{prediction:.2f} kWh")
    m2.metric("Efficiency", f"{efficiency_100km:.1f} kWh/100km")
    
    # use delta for savings
    m3.metric("CO2 Savings vs Petrol", f"{co2_savings:.2f} kg", 
              delta=f"{co2_savings:.2f} kg", delta_color="normal")
    
    if co2_savings < 0:
        st.warning("⚠️ **High Load Alert:** This configuration is currently less CO2-efficient than an average petrol car.")

    # efficiency curve 
    st.divider()
    st.subheader("📈 Efficiency Curve (Physics Wall)")
    with st.spinner("Calculating speed sweeps..."):
        curve_data = plot_efficiency_curve(model, raw_input_df, model_columns, calibration_factor)
        
        # display line chart with explicit axis labels
        st.line_chart(
            curve_data, 
            x_label="Speed (km/h)", 
            y_label="Efficiency (kWh/100km)",
            color="#29b5e8"
        )

    # intelligence & recommendations 
    st.divider()
    st.subheader("💡 Driving Tips")
    tips = []

    if speed > 100:
        tips.append("🚩 **High Speed Drag:** Aerodynamic resistance ($v^2$) is dominating energy drain. Reducing speed by 10km/h could save significant range.")
    elif speed < 30 and speed > 0:
        tips.append("ℹ️ **Low Speed Penalty:** At low speeds, auxiliary loads (like AC) have a higher impact per kilometer.")
    
    if slope > 3:
        tips.append("⛰️ **Incline Work:** Gravity is increasing motor load. Maintain steady throttle to manage battery thermals.")
    elif slope < -2:
        tips.append("🔋 **Regeneration:** Downhill detected. Use regenerative braking to recover energy back to the battery.")

    if temp > 35:
        tips.append("🔥 **Thermal Stress:** High battery temps increase cooling overhead. Efficiency may improve in milder conditions.")
    
    if mode == 3:
        tips.append("🏎️ **Sport Mode Active:** High torque delivery is prioritized over efficiency. Switch to Eco to maximize trip range.")

    if not tips:
        st.success("✅ Operating within optimal efficiency parameters.")
    else:
        for t in tips:
            st.info(t)

else:
    st.info("👈 Adjust the driving conditions in the sidebar and click 'Run Simulation' to see the Digital Twin in action.")