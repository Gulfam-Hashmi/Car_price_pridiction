import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from numpy import sqrt

# --- Configuration (MUST BE THE FIRST Streamlit command) ---
st.set_page_config(layout="wide") 

# --- 1. Load Custom Artifacts and Metrics ---
@st.cache_resource
def load_artifacts():
    try:
        # Load custom JSON files
        with open('preprocessing_config.json', 'r') as f:
            config = json.load(f)
        with open('linear_model_weights.json', 'r') as f:
            weights_data = json.load(f)
            
        # Load data for context (min/max price)
        df = pd.read_csv('cleaned_data_final.csv')
        
    except FileNotFoundError as e:
        # Show specific missing file error
        st.error(f"Missing critical file: {e.filename}. Please ensure all JSON and CSV files from the notebook steps are in this directory.")
        st.stop()

    return config, weights_data, df

config, weights_data, df = load_artifacts()

# Extract components from config and weights
W = np.array(weights_data['weights'])
b = weights_data['bias']
test_r2 = weights_data['test_r2']
test_rmse = weights_data['test_rmse']
scaling_stats = config['scaling_stats']
imputation_stats = config['imputation_stats']
final_columns = config['final_columns']
min_price = df['Cars Prices_Clean'].min()
max_price = df['Cars Prices_Clean'].max()
fuel_types_list = [col.replace('Fuel_Type_Standard_', '') for col in config['categorical_levels']]
numerical_features = ['HorsePower_Clean', 'Total_Speed_Clean', 'Performance_Clean', 'Seats_Clean']


# --- 2. Define Custom Preprocessing/Prediction Functions ---

def standardize_fuel(fuel):
    fuel = str(fuel).lower()
    if 'petrol' in fuel and 'diesel' in fuel:
        return 'Petrol/Diesel'
    elif 'electric' in fuel or 'ev' in fuel:
         return 'Electric'
    elif 'hybrid' in fuel or 'hyrbrid' in fuel or 'plug-in' in fuel:
        return 'Hybrid'
    elif 'diesel' in fuel:
        return 'Diesel'
    elif 'cng' in fuel:
        return 'CNG'
    elif 'hydrogen' in fuel:
        return 'Hydrogen'
    elif 'petrol' in fuel:
        return 'Petrol'
    else:
        return 'Other'

def predict_price(horsepower, total_speed, performance, seats, fuel_type):
    # 1. Create Raw Input DataFrame
    user_input_df = pd.DataFrame({
        'HorsePower_Clean': [horsepower], 
        'Total_Speed_Clean': [total_speed], 
        'Performance_Clean': [performance], 
        'Seats_Clean': [seats], 
        'Fuel_Type_Standard': [standardize_fuel(fuel_type)]
    })
    
    # 2. Manual Preprocessing (Impute, Scale, Encode)
    for col in numerical_features:
        # Impute
        user_input_df[col] = user_input_df[col].fillna(imputation_stats[col])
        # Scale (Z-score)
        mean_val = scaling_stats[col]['mean']
        std_val = scaling_stats[col]['std']
        user_input_df[col] = (user_input_df[col] - mean_val) / std_val
        
    # Encoding (Pandas get_dummies)
    X_encoded = pd.get_dummies(user_input_df, columns=['Fuel_Type_Standard'], dtype=int)
    
    # 3. Align Columns (Crucial for correct matrix multiplication)
    aligned_input = pd.DataFrame(0, index=[0], columns=final_columns)
    for col in X_encoded.columns:
        if col in aligned_input.columns:
            aligned_input[col] = X_encoded[col]

    # 4. Prediction (Y = XW + b) - Pure NumPy
    X_np = aligned_input.values
    prediction = X_np @ W + b
    
    return prediction[0]


# --- 3. Streamlit Application Structure ---

st.title("🚗 Introduction to Data Science Project: Car Price Prediction (NumPy & Pandas)")
st.markdown("---")

st.header("1. Introduction and Project Goals")
st.markdown("""
This project strictly adheres to the constraints of using only **Pandas** (for EDA/Cleaning), **NumPy** (for Machine Learning math), and **Streamlit** (for deployment). 
The model used is a **Linear Regression** implemented via the NumPy Normal Equation.
""")

st.markdown("---")

st.header("2. Exploratory Data Analysis (EDA)")
st.markdown("Visualizations below highlight correlation, feature distribution, and outlier presence (e.g., in car prices).")
col1, col2, col3 = st.columns(3)
with col1:
    try:
        st.image("correlation_heatmap.png", caption="Correlation Matrix")
    except: st.warning("Missing Image: correlation_heatmap.png")
with col2:
    try:
        st.image("horsepower_distribution.png", caption="HorsePower Distribution")
    except: pass
with col3:
    try:
        st.image("outlier_boxplots.png", caption="Price Outliers (Zoomed)")
    except: pass

# Additional EDA table
st.subheader("Feature Counts")
try:
    st.image("top_10_companies.png", caption="Top 10 Car Companies by Count")
except: pass

st.markdown("---")

st.header("3. Model and Runtime Predictions")
st.subheader("Model Performance (Simple Linear Regression)")
st.markdown(f"""
The custom NumPy Linear Regression model was evaluated on unseen test data:

| Metric | Value |
| :--- | :--- |
| **R-squared ($R^2$)** | **{test_r2:.4f}** |
| **Root Mean Squared Error (RMSE)** | **\${test_rmse:,.2f}** |
""")
st.markdown(f"The model explains about **{test_r2 * 100:.2f}\%** of the variance in car prices.")


st.subheader("Interactive Price Prediction")
st.markdown("Input your desired car specifications to predict the price.")

# --- USER INPUT FOR PREDICTION ---
pred_col1, pred_col2 = st.columns(2)
with pred_col1:
    horsepower = st.slider("HorsePower (hp)", min_value=50, max_value=1200, value=300, step=10)
    total_speed = st.slider("Total Speed (km/h)", min_value=100, max_value=400, value=200, step=5)
    
with pred_col2:
    performance = st.slider("0-100 km/h (sec)", min_value=2.0, max_value=20.0, value=7.0, step=0.1)
    seats = st.slider("Seats", min_value=2, max_value=8, value=5, step=1)
    
fuel_type = st.selectbox("Fuel Type", options=fuel_types_list, index=0)


if st.button("Predict Car Price"):
    predicted_price = predict_price(horsepower, total_speed, performance, seats, fuel_type)
    
    # Ensure prediction is non-negative
    predicted_price = max(0, predicted_price)
    
    st.success(f"### Predicted Car Price: **\${predicted_price:,.2f}**")
    
    st.info(f"The dataset price range is from \${min_price:,.2f} to \${max_price:,.2f}.")
    
st.markdown("---")

st.header("4. Conclusion")
st.markdown("""
This project successfully applies a full data science pipeline using only allowed libraries. The process involved manual implementation of scaling, encoding, and a Linear Regression model, demonstrating a deep understanding of the underlying mathematical concepts.
""")