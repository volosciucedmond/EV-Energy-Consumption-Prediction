import pandas as pd  # <--- THIS IS THE MISSING PIECE
import numpy as np

def preprocess_data(df):
    """
    Handles cleaning and categorical encoding.
    """
    # drop identifiers that don't help prediction
    cols_to_drop = ['Vehicle_ID', 'Timestamp', 'Battery_Voltage_V', 'Temperature_C', 'Humidity_%']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # identify categorical columns
    cat_cols = ['Road_Type', 'Traffic_Condition', 'Weather_Condition', 'Driving_Mode']
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df