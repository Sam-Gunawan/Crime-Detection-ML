import streamlit as st
from datetime import date

st.set_page_config(page_title="Crime Detection with GNN", layout="wide")

# Sidebar for user inputs
with st.sidebar:

    # Override the default width for sidebar
    st.markdown(
        """
        <style>
            .stSidebar {
                width: 35% !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("⚙ Model Settings")
    st.markdown("Adjust your inputs here")

    # NOTE: Just placeholders, to be changed according to dataset and model params
    crime_type = st.selectbox("Crime Type", ["All", "Theft", "Assault"])
    some_scale = st.slider("Some numbers", 1, 100, 5)
    location_type = st.radio("Location Type", ["Urban", "Suburban", "Rural"])
    model_choice = st.segmented_control("Choose your model", ["GNN", "Random Forest", "KNN"], selection_mode="single")

    # Date range picker (for temporal ranges)
    date_range = st.date_input("Select timeframe", [date(2005, 1, 1), date(2024, 12, 31)])


# Dashboard
st.header("🧠 AI-Powered Crime Insight")
st.markdown("Visualize and predict urban crime patterns with the latest advancements in technology!")


# Main layout: 2 columns (left: outputs | right: map)
output_col, map_col = st.columns([1, 1.3], gap="medium")

with output_col:
    st.subheader("📊 Model Output")
    st.markdown("*(Put charts and any other evaluation graphics here)")
    
    col1, col2 = st.columns(2, border=True)
    col1.metric("This is cool", "90%", "12%")
    col2.metric("And this", "100", "-🤗")

    with st.expander("Details / Logs"):
        st.text("This can be used to show detailed explanations or logs")

with map_col:
    st.subheader("🗺 Crime Heatmap")
    st.info("Map from pydeck will appear here")


# Footer
st.markdown("---")
st.caption("_Made with ♥ by © 2025 Samuel, Charles, Nisrina, Calvin, and Michael. All rights reserved._")