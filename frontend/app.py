import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="AI Booking Risk", layout="centered")

st.title("🏨 AI Booking Risk Analyzer")

# ----------------------------
# INPUT SECTION
# ----------------------------
st.header("Enter Booking Details")

hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
lead_time = st.slider("Lead Time (days)", 0, 365, 50)

month = st.selectbox("Arrival Month", [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
])

adr = st.number_input("ADR (price per night)", 0.0, 500.0, 100.0)

previous_cancellations = st.number_input("Previous Cancellations", 0, 10, 0)
special_requests = st.number_input("Special Requests", 0, 5, 1)

# ----------------------------
# BUTTON
# ----------------------------
if st.button("Analyze Booking"):

    data = {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_month": month,
        "arrival_date_week_number": 20,
        "arrival_date_day_of_month": 15,
        "meal": "BB",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": 0,
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": adr,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": special_requests,
        "room_mismatch": 0,
        "continent": "Europe",
        "total_guests": 2,
        "total_stay": 3
    }

    try:
        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            result = response.json()

            st.success(f"Risk Level: {result['risk_level']}")
            st.info(f"Confidence: {result['confidence']}%")

            st.subheader("🧠 AI Explanation")
            st.write(result["analysis"])

        else:
            st.error("API error. Check backend.")


    except Exception as e:

        st.error(f"Error: {e}")