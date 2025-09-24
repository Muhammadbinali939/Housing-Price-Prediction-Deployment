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
st.sidebar.image("house gif.jpg", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Predict Price", "ℹ️ About"])

# ---------------- Home Page ----------------
if page == "🏠 Home":
    st.title("🏡 Housing Price Prediction App")
    st.write("Welcome! Predict housing prices based on property features with our interactive app.")

    # Engaging Features
    st.markdown("---")
    st.subheader("✨ Why use this app?")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("💡 Easy Inputs")
        st.write("Adjust features with sliders and inputs.")
    with col2:
        st.info("🚀 Instant Predictions")
        st.write("Get real-time housing price predictions.")
    with col3:
        st.warning("📊 Visual Insights")
        st.write("Interactive charts for confidence ranges and comparisons.")

# ---------------- Prediction Page ----------------
elif page == "📊 Predict Price":
    st.title("📊 Predict Housing Price")

    st.write("Adjust the most influential features below (others set to typical values):")

    # Use expander for mobile-friendly inputs
    with st.expander("Property Features"):
        RM = st.slider("Avg. Rooms per Dwelling (RM)", 3.0, 9.0, float(median_values['RM']))
        LSTAT = st.slider("% Lower Status Population (LSTAT)", 1.0, 40.0, float(median_values['LSTAT']))
        PTRATIO = st.slider("Pupil-Teacher Ratio (PTRATIO)", 10.0, 25.0, float(median_values['PTRATIO']))
        TAX = st.slider("Property Tax Rate (TAX)", 200, 700, int(median_values['TAX']))
        CRIM = st.number_input("Crime Rate (CRIM)", value=float(median_values['CRIM']))
        B = st.number_input("Proportion of Blacks (B)", value=float(median_values['B']))

    st.markdown("---")

    if st.button("🚀 Predict Price"):
        try:
            # Build feature array with user inputs + median defaults
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
            fig.update_layout(
                title_font_size=16,
                xaxis_title_font_size=12,
                yaxis_title_font_size=12,
                margin=dict(l=10, r=10, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------- About Page ----------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    Welcome to the **🏡 Housing Price Prediction App**!  
    This project provides **accurate, real-time housing price predictions** with an intuitive and mobile-friendly interface.
    """)

    st.subheader("🚀 Key Highlights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("💻 Technology Stack")
        st.write("""
        - **Python & Streamlit**  
        - **Scikit-learn**  
        - **Joblib**  
        - **Plotly** for dynamic visualizations
        """)
    with col2:
        st.info("✨ UI & User Experience")
        st.write("""
        - Clean, interactive design  
        - Sliders & inputs for easy feature adjustments  
        - Mobile-friendly layout  
        - Instant results & confidence visualization
        """)
    with col3:
        st.warning("📊 Insights & Analytics")
        st.write("""
        - Confidence range predictions  
        - Interactive bar charts  
        - Helps users make informed decisions
        """)
    
    st.markdown("---")
    st.info("Developed by Muhammad Bin Ali ✨ – Combining AI & Design for smarter real estate insights!")
