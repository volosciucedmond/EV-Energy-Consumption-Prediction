import pandas as pd
import numpy as np

def run_simulation(model, base_data, variable_to_change, new_value, feature_engineer_func):
    """ Simulate how changing one variable affects energy consumption """
    
    # create a copy of the base data
    sim_data = base_data.copy()
    
    # apply the change
    sim_data[variable_to_change] = new_value
    
    # recalculate the physics features based on the new variable value
    sim_data = feature_engineer_func(sim_data)
    
    # ensure columns match the training format
    
    # predict
    prediction = model.predict(sim_data.drop(columns=['Energy_Consumption_Wh']))
    return prediction[0]