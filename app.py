import streamlit as st
import datetime
import pydeck as pdk
import pandas as pd
import joblib
from streamlit_echarts import st_echarts
from streamlit_modal import Modal

st.set_page_config(page_title="Crime Detection with AI", layout="wide")

# Load datasets
df_area = pd.read_csv('./dataset/area_reference.csv')
grouped_week = pd.read_csv('./dataset/grouped_week.csv')
crime_density = pd.read_csv('./dataset/crime_density_by_area.csv')

# Initialize the model and metrics
model = None
results = joblib.load("./models/crime_model_results.pkl")
metrics = {}
for name, result in results.items():
    metrics[name] = {
        'accuracy': result['accuracy'],
        'mae': result['mae'],
        'rmse': result['rmse'],
        'r2': result['r2']
    }

# Sidebar for user inputs
with st.sidebar:

    # Override the default width for sidebar
    st.markdown(
        """
        <style>
            .stSidebar {
                width: 30% !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("⚙ Model Settings")
    st.markdown("Adjust your inputs here")

    # Area selection
    area_name = st.selectbox("Select Area", df_area['AREA NAME'])
    area = df_area[df_area['AREA NAME'] == area_name]['AREA'].values[0]

    # Date selection
    selected_date = st.date_input("Select Date", value=datetime.date.today())
    iso_year, iso_week, _ = selected_date.isocalendar()

    # Get latitude and longitude for the selected area
    area_row = df_area[df_area['AREA'] == area].iloc[0]
    lat = area_row['LAT']
    lon = area_row['LON']

    # Save the selected date and area to a DataFrame for prediction
    input_df = pd.DataFrame({
        'AREA': [area],
        'LAT': [lat],
        'LON': [lon],
        'iso_year': [iso_year],
        'iso_week': [iso_week]
    })

    model_choice = st.segmented_control("Choose your model", ["XGBoost", "Random Forest", "KNN"], selection_mode="single")

    if st.button("Predict"):
        # Load the model based on user choice
        if model_choice == "XGBoost":
            model = joblib.load('./models/xgboost_model.pkl')
        elif model_choice == "Random Forest":
            model = joblib.load('./models/random_forest_model.pkl')
        elif model_choice == "KNN":
            model = joblib.load('./models/knn_model.pkl')
        else:
            st.error("Please select a model to proceed.")


# Dashboard
st.header("🧠 AI-Powered Crime Insight")
st.markdown("Visualize and predict urban crime patterns with the latest advancements in technology!")

# Introduction
with st.expander("What is this app about?", expanded=True):
    st.markdown("""
                    ### 🔍 Crime Detection AI

                    Welcome to **Crime Detection AI** — a data-driven tool designed to estimate the number of weekly crime incidents across various areas using machine learning. Whether you're a policymaker, researcher, or data enthusiast, this platform helps you understand crime patterns with predictive insights.

                    ---

                    ### What is this app about?

                    Our goal is to **forecast weekly crime counts** based on spatial and temporal data, such as:
                    - Area codes
                    - Latitude and longitude
                    - Week and year

                    By identifying trends early, users can better allocate resources, enhance public safety strategies, or simply explore how crimes evolve over time and space.

                    ---

                    ### Models We Use

                    To provide robust predictions, we’ve trained and evaluated **three machine learning models**:

                    - **Random Forest Regressor**  
                    A powerful ensemble model that builds multiple decision trees and averages their predictions for better accuracy and lower overfitting.

                    - **XGBoost Regressor**  
                    An advanced boosting model known for its speed and performance, especially with structured data like time and location.

                    - **K-Nearest Neighbors (KNN)**  
                    A simple yet intuitive model that predicts based on the similarity to nearby data points in time and location.

                    Each model is tested using multiple metrics like MAE, RMSE, R² Score, and a custom accuracy based on real-world tolerance (±20 crimes).

                    ---
                
                    ### Why This Matters

                    Accurate crime prediction can lead to **smarter decision-making**, such as:
                    - Better patrol planning
                    - More efficient policy formulation
                    - Early detection of unusual crime activity

                    Let’s explore the data, understand the patterns, and take a step toward proactive safety.
                """)
    
with st.expander("Learn more about the model evaluation metrics", expanded=False):
        st.markdown("""
                        ### 📊 How We Evaluate Our Models

                        To measure how well each model predicts crime count, we use the following evaluation metrics:

                        ---

                        ### **1. MAE (Mean Absolute Error)**  
                        **What it tells us:**  
                        The average absolute difference between the predicted and actual crime counts.  
                        Lower MAE means the model makes smaller mistakes on average.

                        > **Example:**  
                        If the true crime count was 100 and the model predicted 90, the error is 10. MAE averages these errors across all predictions.

                        ---

                        ### **2. RMSE (Root Mean Squared Error)**  
                        **What it tells us:**  
                        Similar to MAE, but it gives **more weight to larger errors**. It’s the square root of the average of squared differences.  
                        Lower RMSE means more reliable predictions, especially in avoiding big errors.

                        > **Why use it?**  
                        If some predictions are way off, RMSE catches that more than MAE does.

                        ---

                        ### **3. R² Score (Coefficient of Determination)**  
                        **What it tells us:**  
                        A value between 0 and 1 that shows how well the model explains the variation in the data.  
                        - **R² = 1** → perfect predictions  
                        - **R² = 0** → model is no better than guessing the average

                        > **Interpretation:**  
                        A high R² means the model understands the trends in the data well.

                        ---

                        ### **4. Custom Accuracy (Within ±20 Crimes)**  
                        **What it tells us:**  
                        We define accuracy as the **percentage of predictions that fall within ±20 crimes** of the actual value. This is useful because in real-world crime data, small variations are often acceptable.

                        > **Formula:**  
                        ```python
                        accuracy = (abs(y_true - y_pred) <= 20).mean()
                    """)


# Main layout: 2 columns (left: outputs | right: map)
output_col, model_info_col = st.columns([1, 1.3], gap="medium")

with output_col:
    st.subheader("📊 Model Output")

    if model:
        # Make prediction
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Crime Count: {abs(int(prediction/7))}")
        st.info(f"Predicted Crime Count (over a week): {abs(int(prediction))}")

        if st.button("Clear Results"):
            # Clear the input fields
            area_name = None
            selected_date = datetime.date.today()
            model = None
    else:
        st.warning("Please select a model and click 'Predict' to see the results.")


with model_info_col:
    # Model info
    st.subheader("📈 Model Evaluation")

    if model_choice:
        name = model_choice
        model_metrics = metrics[name]

        st.metric("Accuracy (within ±20 crimes)", f"{model_metrics['accuracy']:.2%}", border=True)
        
        option = {
            "title": {
                "text": "",
                "left": "center"
            },
            "tooltip": {},
            "radar": {
                "indicator": [
                    {"name": "MAE", "max": max(1.0, model_metrics['mae'] * 1.5)},
                    {"name": "RMSE", "max": max(1.0, model_metrics['rmse'] * 1.5)},
                    {"name": "R²", "max": 1}
                ]
            },
            "series": [
                {
                    "name": f"{name} Metrics",
                    "type": "radar",
                    "data": [
                        {
                            "value": [model_metrics['mae'], model_metrics['rmse'], model_metrics['r2']],
                            "name": name
                        }
                    ]
                }
            ]
        }

        st_echarts(options=option, height="400px")

    else:
        st.warning("Please select a model to see the evaluation metrics.")


crime_trend_col, heatmap_col = st.columns(2, gap="medium")
with crime_trend_col:
    st.subheader("📊 Weekly Crime Trend")
    area_trend = grouped_week[grouped_week['AREA'] == area].copy()
    area_trend['week_label'] = area_trend['iso_year'].astype(str) + '-W' + area_trend['iso_week'].astype(str).str.zfill(2)
    area_trend = area_trend.sort_values(['iso_year', 'iso_week'])
    st.line_chart(area_trend.set_index('week_label')['crime_count'])

with heatmap_col:
    st.subheader("🗺 Crime Heatmap")
    st.markdown("Visualize the predicted crime density in your area.")
    st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=pdk.ViewState(
                latitude=crime_density['LAT'].mean(),
                longitude=crime_density['LON'].mean(),
                zoom=10,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "HeatmapLayer",
                    data=crime_density,
                    get_position='[LON, LAT]',
                    get_weight='crime_count',
                    radiusPixels=100,
                    opacity=0.7,
                ),
            ],
            height=500
        ))

# Footer
st.markdown("---")
st.caption("_Made with ♥ by © 2025 Samuel, Charles, Nisrina, Calvin, and Michael. All rights reserved._")