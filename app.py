# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# ---------------- Load Model ----------------
model = joblib.load("housing_price_model.pkl")  # trained LinearRegression with 12 features

# Load dataset to get median values
df = pd.read_csv("boston.csv")
median_values = df.median()

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="🏡 Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Sidebar ----------------
st.sidebar.image("house gif.jpg", width=200)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Predict Price", "ℹ️ About"])
# ---------------- Home Page ----------------
if page == "🏠 Home":
    st.title("🏡 Housing Price Prediction App")
    st.write("Welcome! This app helps you **predict housing prices** based on property features.")

    # ----- Engaging Features Section -----
    st.markdown("---")
    st.subheader("✨ Why use this app?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("💡 Easy Inputs")
        st.write("Input your property details with sliders and dropdowns.")
    
    with col2:
        st.info("🚀 Instant Predictions")
        st.write("Get housing price predictions in real-time with our ML model.")
    
    with col3:
        st.warning("📊 Visual Insights")
        st.write("Interactive charts to understand prediction ranges and confidence.")
    
    st.markdown("---")
    st.markdown("🔑 **Features at a glance:**")
    st.markdown("- Beautiful UI with emojis and highlights ✨")
    st.markdown("- Intuitive input options for all users")
    st.markdown("- Predict housing prices instantly")
    st.markdown("- Interactive charts for better insights")


# ---------------- Prediction Page ----------------
elif page == "📊 Predict Price":
    st.title("📊 Predict Housing Price")

    st.write("Adjust the most influential features below (others are set to typical values automatically):")

    # -------- User-friendly Inputs --------
    col1, col2, col3 = st.columns(3)

    with col1:
        RM = st.slider("Avg. Rooms per Dwelling (RM)", 3.0, 9.0, float(median_values['RM']))
        LSTAT = st.slider("% Lower Status Population (LSTAT)", 1.0, 40.0, float(median_values['LSTAT']))
    
    with col2:
        PTRATIO = st.slider("Pupil-Teacher Ratio (PTRATIO)", 10.0, 25.0, float(median_values['PTRATIO']))
        TAX = st.slider("Property Tax Rate (TAX)", 200, 700, int(median_values['TAX']))
    
    with col3:
        CRIM = st.number_input("Crime Rate (CRIM)", value=float(median_values['CRIM']))
        B = st.number_input("Proportion of Blacks (B)", value=float(median_values['B']))

    st.markdown("---")

    if st.button("🚀 Predict Price"):
        # -------- Build feature array with user inputs + median defaults --------
        input_features = np.array([[
            CRIM,
            median_values['ZN'],
            median_values['INDUS'],
            median_values['NOX'],
            RM,
            median_values['AGE'],
            median_values['DIS'],
            median_values['RAD'],
            TAX,
            PTRATIO,
            B,
            LSTAT
        ]])

        # Predict
        prediction = model.predict(input_features)
        prediction_value = float(prediction[0])

        st.success(f"🏠 Estimated Price: ${prediction_value:,.2f}")

        # Confidence Range ±10%
        lower = prediction_value * 0.9
        upper = prediction_value * 1.1
        df_conf = pd.DataFrame({
            "Range": ["Lower", "Prediction", "Upper"],
            "Price": [lower, prediction_value, upper]
        })

        fig = px.bar(
            df_conf, x="Range", y="Price", text="Price",
            color="Range", title="Prediction Confidence Range",
            color_discrete_map={"Lower": "#FFA07A", "Prediction": "#20B2AA", "Upper": "#87CEFA"}
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------- About Page ----------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    Welcome to the **🏡 Housing Price Prediction App**!  
    This project provides **accurate, real-time housing price predictions** with an intuitive and interactive UI.
    """)

    st.subheader("🚀 Key Highlights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("💻 Technology Stack")
        st.write("""
        - **Python & Streamlit** for web app  
        - **Scikit-learn** for ML models  
        - **Joblib** for model saving/loading  
        - **Plotly** for dynamic visualizations
        """)
    with col2:
        st.info("✨ UI & User Experience")
        st.write("""
        - Clean, interactive design  
        - Easy input with sliders & dropdowns  
        - Instant results & confidence visualization  
        - Professional layout with images & icons
        """)
    with col3:
        st.warning("📊 Insights & Analytics")
        st.write("""
        - Confidence range predictions  
        - Interactive bar charts  
        - Quick comparison of property features  
        - Helps users make informed decisions
        """)
    
    st.markdown("---")
    st.info("Developed by Muhammad Bin Ali ✨ – Combining AI & Design for smarter real estate insights!")
