from fastapi import FastAPI
from pydantic import BaseModel
from app.reasoning import analyze_booking

app = FastAPI(title="AI Booking Risk API")

# Input schema
class BookingInput(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    meal: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    booking_changes: int
    deposit_type: str
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    room_mismatch: int
    continent: str
    total_guests: int
    total_stay: int


@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.post("/analyze")
def analyze(data: BookingInput):
    result = analyze_booking(data.dict())
    return result
