import os
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import login


# ================================
# Categorization functions
# ================================
def categorize_lead_time(lead_time: int) -> str:
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
    if nights <= 1:
        return "one-night stay"
    elif nights <= 3:
        return "short stay"
    elif nights <= 7:
        return "week-long stay"
    else:
        return "extended stay"


def categorize_special_requests(count: int) -> str:
    if count == 0:
        return "no special requests"
    elif count == 1:
        return "one special request"
    elif count <= 3:
        return "several special requests"
    else:
        return "many special requests"


def categorize_previous_cancellations(count: int) -> str:
    if count == 0:
        return "no prior cancellations"
    elif count == 1:
        return "one prior cancellation"
    else:
        return "multiple prior cancellations"


def categorize_booking_changes(count: int) -> str:
    if count == 0:
        return "no booking changes"
    elif count == 1:
        return "one booking change"
    else:
        return "multiple booking changes"


def categorize_waiting_list(days: int) -> str:
    if days == 0:
        return "no waiting list time"
    elif days <= 10:
        return "short waiting list"
    else:
        return "long waiting list"


def categorize_cancel_ratio(prev_cancels: int, prev_not_cancel: int) -> str:
    ratio = prev_cancels / (prev_cancels + prev_not_cancel + 1)
    if ratio == 0:
        return "zero cancellation history"
    elif ratio < 0.3:
        return "low historical cancellation rate"
    elif ratio < 0.6:
        return "moderate historical cancellation rate"
    else:
        return "high historical cancellation rate"


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_str(value, default="unknown"):
    try:
        if pd.isna(value):
            return default
        return str(value).strip()
    except Exception:
        return default


# ================================
# Build semantic text for embedding
# ================================
def build_case_text(row: pd.Series) -> str:
    lead_time = safe_int(row.get("lead_time"))
    adr = safe_float(row.get("adr"))
    total_stay = safe_int(row.get("total_stay"))
    special_requests = safe_int(row.get("total_of_special_requests"))
    previous_cancellations = safe_int(row.get("previous_cancellations"))
    booking_changes = safe_int(row.get("booking_changes"))
    waiting_list = safe_int(row.get("days_in_waiting_list", 0))
    previous_not_canceled = safe_int(row.get("previous_bookings_not_canceled", 0))

    deposit_type = safe_str(row.get("deposit_type"))
    market_segment = safe_str(row.get("market_segment"))
    customer_type = safe_str(row.get("customer_type"))
    continent = safe_str(row.get("continent"))
    room_mismatch = safe_int(row.get("room_mismatch"))
    total_guests = safe_int(row.get("total_guests"))
    is_canceled = safe_int(row.get("is_canceled"))

    lead_time_cat = categorize_lead_time(lead_time)
    adr_cat = categorize_adr(adr)
    total_stay_cat = categorize_total_stay(total_stay)
    special_requests_cat = categorize_special_requests(special_requests)
    previous_cancellations_cat = categorize_previous_cancellations(previous_cancellations)
    booking_changes_cat = categorize_booking_changes(booking_changes)
    waiting_list_cat = categorize_waiting_list(waiting_list)
    cancel_ratio_cat = categorize_cancel_ratio(previous_cancellations, previous_not_canceled)

    room_status = "room mismatch exists" if room_mismatch == 1 else "no room mismatch"
    cancel_status = "booking was canceled" if is_canceled == 1 else "booking was not canceled"

    text = (
        f"Hotel booking case. "
        f"This booking has {lead_time_cat}, {adr_cat}, and a {total_stay_cat}. "
        f"There are {total_guests} guests. "
        f"The booking has {special_requests_cat}, {previous_cancellations_cat}, "
        f"{booking_changes_cat}, {waiting_list_cat}, and {cancel_ratio_cat}. "
        f"Deposit type is {deposit_type}. "
        f"Market segment is {market_segment}. "
        f"Customer type is {customer_type}. "
        f"Guest region is {continent}. "
        f"The case has {room_status}. "
        f"Final outcome: {cancel_status}."
    )

    return text


# ================================
# Main
# ================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")

if HF_TOKEN:
    login(token=HF_TOKEN)

# Load dataset
df = pd.read_csv("../Data/cases.csv")

# Validate required columns
required_columns = [
    "case_id", "is_canceled", "lead_time", "deposit_type", "market_segment",
    "customer_type", "previous_cancellations", "booking_changes",
    "total_of_special_requests", "adr", "total_guests", "total_stay",
    "continent", "room_mismatch"
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in CSV: {missing_columns}")

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "booking-decision-index"

# Delete old index if it exists
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    time.sleep(5)

# Create new index
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

time.sleep(5)
index = pc.Index(index_name)

# ================================
# Prepare and upload vectors
# ================================
batch_size = 50
batch = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Uploading cases"):
    case_id = str(safe_int(row["case_id"]))

    # Build semantic text using categories
    semantic_text = build_case_text(row)

    # Create embedding from semantic text
    embedding = model.encode(semantic_text).tolist()

    lead_time = safe_int(row.get("lead_time"))
    adr = safe_float(row.get("adr"))
    total_stay = safe_int(row.get("total_stay"))
    special_requests = safe_int(row.get("total_of_special_requests"))
    previous_cancellations = safe_int(row.get("previous_cancellations"))
    booking_changes = safe_int(row.get("booking_changes"))
    waiting_list = safe_int(row.get("days_in_waiting_list", 0))
    previous_not_canceled = safe_int(row.get("previous_bookings_not_canceled", 0))

    # Store both raw and categorized info
    metadata = {
        "text": semantic_text,  # important for LangChain retrieval
        "case_id": safe_int(row["case_id"]),
        "is_canceled": safe_int(row["is_canceled"]),

        # raw values
        "lead_time": lead_time,
        "adr": adr,
        "total_stay": total_stay,
        "total_guests": safe_int(row["total_guests"]),
        "previous_cancellations": previous_cancellations,
        "booking_changes": booking_changes,
        "total_of_special_requests": special_requests,
        "days_in_waiting_list": waiting_list,
        "previous_bookings_not_canceled": previous_not_canceled,
        "deposit_type": safe_str(row["deposit_type"]),
        "market_segment": safe_str(row["market_segment"]),
        "customer_type": safe_str(row["customer_type"]),
        "continent": safe_str(row["continent"]),
        "room_mismatch": safe_int(row["room_mismatch"]),

        # categorized values
        "lead_time_category": categorize_lead_time(lead_time),
        "adr_category": categorize_adr(adr),
        "total_stay_category": categorize_total_stay(total_stay),
        "special_requests_category": categorize_special_requests(special_requests),
        "previous_cancellations_category": categorize_previous_cancellations(previous_cancellations),
        "booking_changes_category": categorize_booking_changes(booking_changes),
        "waiting_list_category": categorize_waiting_list(waiting_list),
        "cancel_ratio_category": categorize_cancel_ratio(previous_cancellations, previous_not_canceled),
    }

    batch.append({
        "id": case_id,
        "values": embedding,
        "metadata": metadata
    })

    if len(batch) == batch_size:
        index.upsert(vectors=batch)
        batch = []

if batch:
    index.upsert(vectors=batch)

print("All data uploaded to Pinecone successfully!")
