import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="AI Booking Risk", layout="centered")

st.title("🏨 AI Booking Risk Analyzer")

# ----------------------------
# BASIC INPUTS
# ----------------------------
st.header("Enter Booking Details")

hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
lead_time = st.slider("Lead Time (days)", 0, 365, 50)

month = st.selectbox(
    "Arrival Month",
    [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
)

adr = st.number_input("ADR (price per night)", 0.0, 500.0, 100.0)

previous_cancellations = st.number_input("Previous Cancellations", 0, 10, 0)
special_requests = st.number_input("Special Requests", 0, 5, 1)

total_guests = st.number_input("Total Guests", 1, 10, 2)
total_stay = st.number_input("Total Stay (nights)", 1, 30, 3)

# ----------------------------
# ADVANCED INPUTS
# ----------------------------
with st.expander("Advanced Options"):

    meal = st.selectbox("Meal Type", ["BB", "HB", "FB", "SC"])

    market_segment = st.selectbox(
        "Market Segment",
        ["Online TA", "Direct", "Corporate"]
    )

    distribution_channel = st.selectbox(
        "Distribution Channel",
        ["TA/TO", "Direct", "Corporate"]
    )

    is_repeated_guest = st.selectbox("Repeated Guest?", [0, 1])

    previous_bookings_not_canceled = st.number_input(
        "Previous Successful Bookings", 0, 10, 0
    )

    booking_changes = st.number_input("Booking Changes", 0, 10, 0)

    deposit_type = st.selectbox(
        "Deposit Type",
        ["No Deposit", "Non Refund", "Refundable"]
    )

    customer_type = st.selectbox(
        "Customer Type",
        ["Transient", "Transient-Party", "Contract", "Group"]
    )

    continent = st.selectbox(
        "Continent",
        ["Europe", "Asia", "Africa", "Other"]
    )

    parking = st.number_input("Parking Spaces", 0, 5, 0)

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
        "meal": meal,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "days_in_waiting_list": 0,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": parking,
        "total_of_special_requests": special_requests,
        "room_mismatch": 0,
        "continent": continent,
        "total_guests": total_guests,
        "total_stay": total_stay
    }

    try:
        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            result = response.json()

            st.success(f"Risk Level: {result['risk_level']}")
            st.info(f"Confidence: {result['confidence']}%")

            st.subheader("📊 Similar Past Cases")
            st.text(result["retrieved_cases"])

            st.subheader("🧠 AI Explanation")
            st.write(result["analysis"])

        else:
            st.error("API error. Check backend.")

    except Exception as e:
        st.error(f"Error: {e}")
