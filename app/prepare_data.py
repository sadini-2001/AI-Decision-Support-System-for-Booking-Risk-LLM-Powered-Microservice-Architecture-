import pandas as pd

# Load dataset
df = pd.read_csv("../Data/hotel_booking_cancellation.csv")  # change name if needed

# Create case_id
df = df.reset_index(drop=True)
df["case_id"] = df.index

# Convert row → case text
def yes_no(val):
    return "Yes" if val == 1 else "No"

def cancel_label(val):
    return "Canceled" if val == 1 else "Not Canceled"

def row_to_case(row):
    return f"""
Booking case:

Hotel type: {row['hotel']}

Arrival details:
Month: {row['arrival_date_month']}
Week number: {row['arrival_date_week_number']}
Day of month: {row['arrival_date_day_of_month']}
Lead time: {row['lead_time']} days

Booking details:
Meal plan: {row['meal']}
Market segment: {row['market_segment']}
Distribution channel: {row['distribution_channel']}
Deposit type: {row['deposit_type']}
Customer type: {row['customer_type']}

Customer behavior:
Repeated guest: {yes_no(row['is_repeated_guest'])}
Previous cancellations: {row['previous_cancellations']}
Previous bookings not canceled: {row['previous_bookings_not_canceled']}

Booking activity:
Booking changes: {row['booking_changes']}
Days in waiting list: {row['days_in_waiting_list']}
Special requests: {row['total_of_special_requests']}

Stay & financial:
ADR: {row['adr']}
Total guests: {row['total_guests']}
Total stay: {row['total_stay']} nights
Required parking spaces: {row['required_car_parking_spaces']}

Other:
Room mismatch: {yes_no(row['room_mismatch'])}
Continent: {row['continent']}

Outcome: {cancel_label(row['is_canceled'])}
""".strip()

# Apply function
df["case_text"] = df.apply(row_to_case, axis=1)

# Save processed dataset
df.to_csv("../Data/cases.csv", index=False)

# Quick check
print("Sample case:\n")
print(df["case_text"].iloc[0])

print("\nTotal cases:", len(df))
print("File saved at: Data/cases.csv")