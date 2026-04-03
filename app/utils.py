def yes_no(val):
    return "Yes" if int(val) == 1 else "No"


def cancel_label(val):
    return "Canceled" if int(val) == 1 else "Not Canceled"

# For dataset rows that already have is_canceled
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

# For dataset rows that already have is_canceled
def input_to_case(data: dict):
    return f"""
Booking case:

Hotel type: {data['hotel']}

Arrival details:
Month: {data['arrival_date_month']}
Week number: {data['arrival_date_week_number']}
Day of month: {data['arrival_date_day_of_month']}
Lead time: {data['lead_time']} days

Booking details:
Meal plan: {data['meal']}
Market segment: {data['market_segment']}
Distribution channel: {data['distribution_channel']}
Deposit type: {data['deposit_type']}
Customer type: {data['customer_type']}

Customer behavior:
Repeated guest: {yes_no(data['is_repeated_guest'])}
Previous cancellations: {data['previous_cancellations']}
Previous bookings not canceled: {data['previous_bookings_not_canceled']}

Booking activity:
Booking changes: {data['booking_changes']}
Days in waiting list: {data['days_in_waiting_list']}
Special requests: {data['total_of_special_requests']}

Stay & financial:
ADR: {data['adr']}
Total guests: {data['total_guests']}
Total stay: {data['total_stay']} nights
Required parking spaces: {data['required_car_parking_spaces']}

Other:
Room mismatch: {yes_no(data['room_mismatch'])}
Continent: {data['continent']}
""".strip()