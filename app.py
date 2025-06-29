# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
from rag_chatbot import query_rag  # <- Chatbot logic from rag_chatbot.py

# Load the trained model
model = joblib.load("resampled_customer_value_model.joblib") 

# Set page config
st.set_page_config(page_title="Full-Stack ML + RAG App", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Controls")

tab1, tab2 = st.tabs(["📊 Real-Time Dashboard", "💬 RAG Chatbot"])

# ---------------- TAB 1 ------------------
with tab1:
    st.header("📊 Real-Time Customer Value Prediction")

    # Simulation meta
    products = {
        "Laptop": (800, 1500),
        "Smartphone": (300, 1000),
        "Monitor": (150, 500),
        "Headphones": (20, 200),
        "Keyboard": (30, 150)
    }
    regions = ["North", "South", "East", "West"]
    device_types = ["Mobile", "Desktop", "Tablet"]
    speed = st.sidebar.slider("Simulation Speed (sec)", 0.5, 5.0, 1.0, 0.5)
    selected_products = st.sidebar.multiselect("Filter by Product", list(products.keys()), default=list(products.keys()))
    max_rows = st.sidebar.number_input("Max rows to keep in table", 50, 500, 100)
    simulate = st.sidebar.toggle("Start Simulation", value=False)

    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame()

    def simulate_entry():
        product = random.choice(list(products.keys()))
        price = round(np.random.normal(*products[product]), 2)
        clicks = np.random.poisson(8)
        region = random.choice(regions)
        device = random.choices(device_types, weights=[0.6, 0.3, 0.1])[0]
        user_age = int(np.clip(np.random.normal(35, 12), 18, 70))
        session_time = max(1, np.random.exponential(scale=10))
        is_returning_user = random.choices([True, False], weights=[0.4, 0.6])[0]
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "product": product,
            "price": max(10, price),
            "clicks": max(0, clicks),
            "region": region,
            "device_type": device,
            "user_age": user_age,
            "session_time": round(session_time, 2),
            "is_returning_user": is_returning_user
        }

    table_area = st.empty()
    plot_area = st.empty()

    if simulate:
        while True:
            entry = simulate_entry()
            entry_df = pd.DataFrame([entry])
            X = entry_df.drop(columns=["timestamp", "product"])
            prediction = model.predict(X)[0]
            entry_df["prediction"] = prediction
            st.session_state.data = pd.concat([st.session_state.data, entry_df], ignore_index=True).tail(max_rows)

            filtered_df = st.session_state.data[st.session_state.data["product"].isin(selected_products)]

            table_area.subheader("Live Data with Predictions")
            table_area.dataframe(filtered_df, use_container_width=True)

            plot_data = filtered_df.groupby(["timestamp", "prediction"]).size().unstack(fill_value=0)
            if not plot_data.empty:
                fig, ax = plt.subplots()
                plot_data.plot(kind='bar', stacked=True, ax=ax, figsize=(12, 5), colormap="Set2")
                ax.set_title("Prediction Counts Over Time")
                ax.set_ylabel("Count")
                ax.set_xlabel("Timestamp")
                plt.xticks(rotation=45)
                plot_area.pyplot(fig)

            time.sleep(speed)
    else:
        st.info("🟡 Toggle 'Start Simulation' in the sidebar to begin real-time prediction.")

# ---------------- TAB 2 ------------------
with tab2:
    st.header("🤖 Code Companion: Chat with Python / ML / SQL Docs")

    query = st.text_input("Ask a question about Python / ML / SQL docs")
    if st.button("Submit") and query:
        with st.spinner("Searching documents..."):
            answer = query_rag(query)
            st.success(answer)
