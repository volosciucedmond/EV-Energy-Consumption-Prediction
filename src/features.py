import pandas as pd
import numpy as np

def create_physics_features(df):
    """ For physics based proxies """
    
    # convert speed to m/s
    df['Speed_ms'] = df['Speed_kmh'] / 3.6
    
    # aerodynamic drag (proportional to v^2 or v^3)
    df["Drag_Proxy"] = df["Speed_ms"] ** 2 
    df["Aero_Power_Demand"] = df["Speed_ms"] ** 3  
    
    # proportional energy / incline load
    # force = mass * g * sin(theta)
    
    df["Incline_Work_Proxy"] = df["Vehicle_Weight_kg"] * df["Speed_ms"] * (df["Slope_%"] / 100)
    
    # thermal stress index
    df["Thermal_Stress_Index"] = df["Battery_Temperature_C"] * df["Speed_ms"]
    
    # kinetic energy proxy
    df["Kinetic_Energy_Proxy"] = 0.5 * df["Vehicle_Weight_kg"] * (df["Speed_ms"] ** 2)
    
    return df

def calculate_efficiency_metrics(df):
    """ Calculates standard ev efficiency metrics """
    
    # avoid division by zero for stationary points:
    df = df[df['Distance_Travelled_km'] > 0].copy()
    
    #kWh per 100km
    df['kWh_per_100km'] = (df['Energy_Consumption_kWh'] / df['Distance_Travelled_km']) * 100
    
    # Wh per km
    df['Wh_per_km'] = (df['Energy_Consumption_kWh'] / df['Distance_Travelled_km']) * 1000
    
    return df

def calculate_carbon_footprint(df, carbon_intensity=0.4):
    """
    Calculates CO2 emissions based on energy consumption.
    carbon_intensity: kg of CO2 per kWh (0.4 is a rough global average).
    """
    # kg of CO2 produced for the trip
    df['CO2_emitted_kg'] = df['Energy_Consumption_kWh'] * carbon_intensity
    
    # kg of CO2 per 100km
    df['CO2_per_100km'] = (df['CO2_emitted_kg'] / df['Distance_Travelled_km']) * 100
    
    return df