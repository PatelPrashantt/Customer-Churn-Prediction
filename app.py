import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model.pkl")
accuracy = joblib.load("accuracy.pkl")

st.title("📊 Customer Churn Prediction")

# ---------------- ACCURACY ----------------
st.subheader("📈 Model Accuracy")
st.write(f"{round(accuracy*100,2)}%")

st.markdown("---")

# ---------------- INPUT ----------------
st.subheader("🔮 Enter Customer Details")

tenure = st.slider("Tenure (months)", 0, 72)
monthly = st.number_input("Monthly Charges", value=50.0)

# ✅ AUTO CALCULATE TOTAL CHARGES
total = tenure * monthly
st.write(f"💰 Total Charges (Auto Calculated): {total}")

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    prediction = model.predict(input_data)

    # ---------------- RESULT ----------------
    if prediction[0] == 1:
        st.error("⚠️ Customer will churn")
        churn_value = 1
        stay_value = 0
    else:
        st.success("✅ Customer will stay")
        churn_value = 0
        stay_value = 1

    # ---------------- DYNAMIC GRAPH ----------------
    st.subheader("📊 Prediction Graph")

    fig, ax = plt.subplots()
    ax.bar(["Churn", "Stay"], [churn_value, stay_value])
    ax.set_title("Customer Prediction Result")
    st.pyplot(fig)
    