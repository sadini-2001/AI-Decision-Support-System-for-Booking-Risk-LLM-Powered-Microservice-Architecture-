import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from utils import input_to_case

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Check your .env file.")

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("booking-decision-index")


# Retrieval function
def retrieve_similar_cases(case_data: dict, top_k: int = 10, threshold: float = 0.75):
    # Convert input into case text
    case_text = input_to_case(case_data)

    # Convert into embedding
    query_embedding = model.encode(case_text).tolist()

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    matches = results["matches"]

    # Apply threshold filtering
    filtered = [
        m for m in matches
        if m["score"] >= threshold
    ]

    # Fallback
    if len(filtered) == 0:
        print(" No matches above threshold, using top 3 fallback")
        filtered = matches[:3]

    return case_text, filtered


# Pretty print function
def display_results(case_text, matches):
    print("\nGenerated Case:\n")
    print(case_text)

    print("\nSimilar Cases:\n")

    for i, match in enumerate(matches, start=1):
        meta = match["metadata"]

        print(f"Match {i}")
        print(f"Score: {match['score']:.4f}")
        print(f"Case ID: {meta.get('case_id')}")
        print(f"Canceled: {meta.get('is_canceled')}")
        print(f"Lead Time: {meta.get('lead_time')}")
        print(f"Deposit Type: {meta.get('deposit_type')}")
        print(f"Market Segment: {meta.get('market_segment')}")
        print("-" * 40)


# Testing
if __name__ == "__main__":
    sample_case = {
        "hotel": "City Hotel",
        "lead_time": 120,
        "arrival_date_month": "July",
        "arrival_date_week_number": 28,
        "arrival_date_day_of_month": 15,
        "meal": "BB",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 1,
        "previous_bookings_not_canceled": 0,
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 95.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
        "room_mismatch": 1,
        "continent": "Europe",
        "total_guests": 2,
        "total_stay": 3
    }

    case_text, matches = retrieve_similar_cases(sample_case)

    display_results(case_text, matches)