🚗 EV Energy Digital Twin & Simulator
=====================================

🚀 Live Demo: [Check out the EV Digital Twin Simulator here](https://ev-energy-consumption-prediction-aymhkdwbg4v635vhmjajpl.streamlit.app/)


A physics-informed machine learning pipeline designed to predict battery electric vehicle (BEV) energy consumption ($kWh$) and efficiency ($kWh/100km$) based on real-world driving conditions.

This project features a modular backend, a comparative model study, and an interactive **Streamlit** dashboard for real-time trip simulation.

📌 Project Overview
-------------------

The goal was to build a "digital twin" that understands how speed, road incline, and temperature impact battery drain. Unlike standard black-box models, this system prioritizes **physical consistency**, ensuring that predictions align with the laws of aerodynamics and thermodynamics.

🛠️ Technical Implementation
----------------------------

### 1\. Physics-Informed Feature Engineering

To ensure the model understands why energy is consumed, raw data was transformed into physical proxies:

*   **Drag proxy ($v^2$):** Models aerodynamic resistance, which increases quadratically with velocity.
    
*   **Incline work ($slope * weight * v$):** Models the gravitational potential energy required to climb slopes.
    
*   **Thermal stress:** An interaction term between battery temperature and speed to account for cooling/heating overhead.
    

### 2\. Model Selection: Robustness over Accuracy

We conducted a head-to-head comparison between **Linear Regression** and **XGBoost**.

| Metric           | Linear Regression (champion) | XGBoost (baseline) |   |   |
|------------------|------------------------------|--------------------|---|---|
| R² Score         | 0.887                        | 0.895              |   |   |
| MAE              | 0.602 kWh                    | 0.573 kWh          |   |   |
| Interpretability | High (physics aligned)       | Low (black box)    |   |   |

**Decision:** Although XGBoost showed a ~0.8% higher accuracy, **Linear Regression** was selected as the champion model. It provides stable extrapolation and ensures that energy consumption always increases with speed and incline, preventing the "unphysical" predictions sometimes found in tree-based models on small datasets.

⚠️ Honest Reflection: The "Data-Reality" Gap
--------------------------------------------

During validation, I identified that the training dataset (average efficiency ~60-80 kWh/100km) represented energy consumption roughly **4x higher** than a standard consumer BEV like a Tesla Model 3 or VW ID.3.

**The Solution:**

*   The **model** remains honest to the raw data (it is a high-performance/heavy-duty model).
    
*   The **UI** includes a **0.25x calibration factor** toggle. This allows the simulator to provide realistic "passenger BEV" results (~15-20 kWh/100km) while maintaining the physics-based relationships learned from the data.
    

🖥️ Interactive Simulator
-------------------------

The project includes a **Streamlit** dashboard that allows users to:

*   Adjust speed, weight, incline, and temperature via sliders.
    
*   Get real-time feedback on trip energy and $CO\_2$ impact.
    
*   Receive **Context-Aware Driving Tips** based on their current "driving" style (e.g., aerodynamic advice for high speeds or regen tips for downhills).
    

📂 Project Structure
--------------------

*   src/: Core logic (preprocessing, physics engineering, evaluation).
    
*   notebooks/: Modular workflow (01\_EDA, 02\_Training, 03\_Simulation).
    
*   models/: Saved .pkl files for the model and column transformers.
    
*   app.py: The streamlit dashboard source code.
    

🚀 How to Run
-------------
**Option 1:**
Via [Streamlit](https://ev-energy-consumption-prediction-aymhkdwbg4v635vhmjajpl.streamlit.app/)

**Option 2:**

1.  Clone the repo.
    
2.  Install dependencies: pip install -r requirements.txt.
    
3.  Run the UI: streamlit run app.py.

