# ============================================
# utils.py
# Common helper functions used across the project
# ============================================


# ============================================
# BASIC HELPER FUNCTIONS
# ============================================
def yes_no(val):
    """
    Convert 0/1 type value into Yes/No.
    """
    return "Yes" if int(val) == 1 else "No"


def cancel_label(val):
    """
    Convert cancellation flag into readable label.
    """
    return "Canceled" if int(val) == 1 else "Not Canceled"

def input_to_case(data: dict):
    """
    Convert user input data into readable booking case text.
    No outcome is included because this is used before prediction.
    """
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


# ============================================
# CATEGORIZATION FUNCTIONS
# Used to convert raw numeric values into semantic labels
# for Pinecone retrieval and LLM-friendly reasoning
# ============================================
def categorize_lead_time(lead_time: int) -> str:
    """
    Convert lead time into a semantic category.
    """
    if lead_time <= 7:
        return "very short lead time"
    elif lead_time <= 30:
        return "short lead time"
    elif lead_time <= 90:
        return "moderate lead time"
    elif lead_time <= 180:
        return "long lead time"
    else:
        return "very long lead time"


def categorize_adr(adr: float) -> str:
    """
    Convert ADR into a price category.
    """
    if adr < 50:
        return "very low rate"
    elif adr < 100:
        return "budget rate"
    elif adr < 150:
        return "mid-range rate"
    elif adr < 250:
        return "premium rate"
    else:
        return "luxury rate"


def categorize_total_stay(nights: int) -> str:
    """
    Convert total stay nights into a stay-length category.
    """
    if nights <= 1:
        return "one-night stay"
    elif nights <= 3:
        return "short stay"
    elif nights <= 7:
        return "week-long stay"
    else:
        return "extended stay"


def categorize_special_requests(count: int) -> str:
    """
    Convert number of special requests into a category.
    """
    if count == 0:
        return "no special requests"
    elif count == 1:
        return "one special request"
    elif count <= 3:
        return "several special requests"
    else:
        return "many special requests"


def categorize_previous_cancellations(count: int) -> str:
    """
    Convert previous cancellations count into a category.
    """
    if count == 0:
        return "no prior cancellations"
    elif count == 1:
        return "one prior cancellation"
    else:
        return "multiple prior cancellations"


def categorize_booking_changes(count: int) -> str:
    """
    Convert booking changes count into a category.
    """
    if count == 0:
        return "no booking changes"
    elif count == 1:
        return "one booking change"
    else:
        return "multiple booking changes"


def categorize_waiting_list(days: int) -> str:
    """
    Convert waiting list days into a category.
    """
    if days == 0:
        return "no waiting list time"
    elif days <= 10:
        return "short waiting list"
    else:
        return "long waiting list"


def categorize_cancel_ratio(prev_cancels: int, prev_not_cancel: int) -> str:
    """
    Convert cancellation ratio into a category.
    """
    ratio = prev_cancels / (prev_cancels + prev_not_cancel + 1)

    if ratio == 0:
        return "zero cancellation history"
    elif ratio < 0.3:
        return "low historical cancellation rate"
    elif ratio < 0.6:
        return "moderate historical cancellation rate"
    else:
        return "high historical cancellation rate"
