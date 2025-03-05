import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Streamlit page config
st.set_page_config(page_title="BAI Fitness Tracker", page_icon="🏋️", layout="wide")

# Custom styles
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 10px;
        }
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Sidebar: User Input ----
st.sidebar.header("🏃‍♂️ Log Your Activity")
activity = st.sidebar.selectbox("Select Activity", ["Running", "Cycling", "Swimming", "Walking", "Gym Workout"])
duration = st.sidebar.slider("Duration (minutes)", min_value=5, max_value=180, step=5, value=30)
weight = st.sidebar.number_input("Your Weight (kg)", min_value=30, max_value=150, step=1, value=70)
date = st.sidebar.date_input("Date", datetime.date.today())

# ---- ML Model Training ----
# Sample dataset
data = {
    "Activity": ["Running", "Cycling", "Swimming", "Walking", "Gym Workout"] * 50,
    "Duration": np.random.randint(10, 120, 250),
    "Weight": np.random.randint(50, 100, 250),
    "Calories Burned": np.random.randint(100, 900, 250)
}

df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=["Activity"], drop_first=True)

# Train ML model
X = df.drop("Calories Burned", axis=1)
y = df["Calories Burned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save & Load Model
with open("calories_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("calories_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Convert user input for ML prediction
user_input = pd.DataFrame([[duration, weight] + [1 if activity == act else 0 for act in ["Cycling", "Swimming", "Walking", "Gym Workout"]]],
                          columns=X.columns)

predicted_calories = loaded_model.predict(user_input)[0]

# ---- Main Content ----
st.markdown("## 🏋️ BAI Fitness Tracker")
st.write("Track your fitness progress with real-time analytics & AI-powered calorie estimation.")

# Predicted Calories Burned
st.metric("🔥 Estimated Calories Burned", f"{predicted_calories:.2f} kcal")

# Save activity log
if st.sidebar.button("📌 Add Entry"):
    entry = {"Activity": activity, "Duration (mins)": duration, "Weight (kg)": weight, "Calories Burned": predicted_calories, "Date": date}
    
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.append(entry)
    st.sidebar.success("Activity Logged Successfully! ✅")

# ---- Display Activity Log ----
st.subheader("📅 Your Activity Log")
if "logs" in st.session_state and st.session_state.logs:
    df_log = pd.DataFrame(st.session_state.logs)
    st.dataframe(df_log)
else:
    st.info("No activities logged yet. Start by adding an entry!")

# ---- Analytics & Visualization ----
st.subheader("📊 Fitness Progress")

if "logs" in st.session_state and st.session_state.logs:
    df_chart = pd.DataFrame(st.session_state.logs)

    # Calories Burned Over Time
    fig1 = px.line(df_chart, x="Date", y="Calories Burned", title="🔥 Calories Burned Over Time", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    # Activity Distribution
    fig2 = px.pie(df_chart, names="Activity", values="Duration (mins)", title="🏃‍♂️ Activity Distribution", hole=0.4)
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.warning("No data to visualize. Start logging your activities!")

# Run the app using: streamlit run app.py
