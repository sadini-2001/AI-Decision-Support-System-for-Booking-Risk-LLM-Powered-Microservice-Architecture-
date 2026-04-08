import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Any

from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import LLM

from app.utils import input_to_case

# ─────────────────────────────────────────────
# 1. ENVIRONMENT & API KEYS
# ─────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")


# ─────────────────────────────────────────────
# 2. LOAD TRAINED RANDOM FOREST MODEL
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

try:
    rf_model = joblib.load(MODEL_PATH)
    print(f"[INFO] Random Forest model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Make sure model/model.pkl exists."
    )


# ─────────────────────────────────────────────
# 3. FEATURE CATEGORIZATION FUNCTIONS
#    Used for BOTH embedding text AND retrieved
#    case formatting → consistent language,
#    no raw numbers passed to LLM
# ─────────────────────────────────────────────
MONTHS = ['January','February','March','April','May','June',
          'July','August','September','October','November','December']

ONE_HOT_COLS = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "meal": ["BB", "FB", "HB", "SC", "Undefined"],
    "market_segment": ["Aviation", "Complementary", "Corporate",
                       "Direct", "Groups", "Offline TA/TO",
                       "Online TA", "Undefined"],
    "distribution_channel": ["Corporate", "Direct", "GDS",
                             "TA/TO", "Undefined"],
    "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
    "customer_type": ["Contract", "Group", "Transient", "Transient-Party"],
    "continent": ["Africa", "Americas", "Asia", "Europe",
                  "Oceania", "Others"]
}

def categorize_lead_time(lead_time: int) -> str:
    if lead_time <= 7:
        return "very short (same week)"
    elif lead_time <= 30:
        return "short (within a month)"
    elif lead_time <= 90:
        return "moderate (1-3 months)"
    elif lead_time <= 180:
        return "long (3-6 months)"
    else:
        return "very long (over 6 months)"

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
        return "one night"
    elif nights <= 3:
        return "short (2-3 nights)"
    elif nights <= 7:
        return "week-long"
    else:
        return "extended (over a week)"

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
        return f"{count} prior cancellations"


# ─────────────────────────────────────────────
# 4. FEATURE EXTRACTION FOR ML MODEL
# ─────────────────────────────────────────────
def extract_ml_features(case_data: dict) -> pd.DataFrame:
    prev_cancels     = case_data.get("previous_cancellations", 0)
    prev_not_cancel  = case_data.get("previous_bookings_not_canceled", 0)
    lead_time        = case_data.get("lead_time", 0)

    numerical = {
        "lead_time":                      lead_time,
        "arrival_date_month":             MONTHS.index(case_data.get("arrival_date_month", "January")) + 1,
        "arrival_date_week_number":       case_data.get("arrival_date_week_number", 0),
        "arrival_date_day_of_month":      case_data.get("arrival_date_day_of_month", 0),
        "is_repeated_guest":              case_data.get("is_repeated_guest", 0),
        "previous_cancellations":         prev_cancels,
        "previous_bookings_not_canceled": prev_not_cancel,
        "booking_changes":                case_data.get("booking_changes", 0),
        "days_in_waiting_list":           case_data.get("days_in_waiting_list", 0),
        "adr":                            case_data.get("adr", 0.0),
        "required_car_parking_spaces":    case_data.get("required_car_parking_spaces", 0),
        "total_of_special_requests":      case_data.get("total_of_special_requests", 0),
        "room_mismatch":                  case_data.get("room_mismatch", 0),
        "total_guests":                   case_data.get("total_guests", 1),
        "total_stay":                     case_data.get("total_stay", 1),
        # engineered features
        "high_risk_flag": int(prev_cancels > 0 and lead_time > 100),
        "cancel_ratio":   prev_cancels / (prev_cancels + prev_not_cancel + 1),
    }

    one_hot = {}
    for original_col, categories in ONE_HOT_COLS.items():
        value = case_data.get(original_col, "")
        for category in categories:
            one_hot[f"{original_col}_{category}"] = 1 if value == category else 0

    df = pd.DataFrame([{**numerical, **one_hot}])
    df = df.reindex(columns=rf_model.feature_names_in_, fill_value=0)
    return df


# ─────────────────────────────────────────────
# 5. PINECONE VECTOR STORE + RETRIEVER
#    FIX 5: score_threshold raised to 0.65
#    for cleaner, more relevant results
# ─────────────────────────────────────────────
pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("booking-decision-index")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.65}   # FIX 5: raised from 0.6
)


# ─────────────────────────────────────────────
# 6. GROQ LLM WRAPPER
# ─────────────────────────────────────────────
_groq_client = Groq(api_key=GROQ_API_KEY)

class GroqLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3
        )
        return response.choices[0].message.content

llm = GroqLLM()


# ─────────────────────────────────────────────
# 7. CONVERT FEATURES TO EMBEDDING TEXT
# ─────────────────────────────────────────────
def convert_features_to_text(case_data: dict) -> str:
    hotel          = case_data.get("hotel", "Unknown")
    month          = case_data.get("arrival_date_month", "Unknown")
    meal           = case_data.get("meal", "Unknown")
    market         = case_data.get("market_segment", "Unknown")
    distribution   = case_data.get("distribution_channel", "Unknown")
    deposit        = case_data.get("deposit_type", "Unknown")
    customer_type  = case_data.get("customer_type", "Unknown")
    repeated_guest = "returning guest" if case_data.get("is_repeated_guest", 0) == 1 else "new guest"
    room_mismatch  = "room was changed" if case_data.get("room_mismatch", 0) == 1 else "room as booked"
    continent      = case_data.get("continent", "Unknown")
    total_guests   = case_data.get("total_guests", 1)

    return (
        f"{hotel} booking for {total_guests} guest(s) arriving in {month} "
        f"from {continent}, {repeated_guest}, "
        f"lead time {categorize_lead_time(case_data.get('lead_time', 0))}, "
        f"{meal} meal plan, {market} market via {distribution} channel, "
        f"{deposit} deposit, {customer_type} customer, "
        f"{categorize_previous_cancellations(case_data.get('previous_cancellations', 0))}, "
        f"{categorize_special_requests(case_data.get('total_of_special_requests', 0))}, "
        f"stay {categorize_total_stay(case_data.get('total_stay', 0))}, "
        f"{categorize_adr(case_data.get('adr', 0))}, {room_mismatch}."
    )


# ─────────────────────────────────────────────
# 8. ML-BASED RISK PREDICTION
#    FIX 3: probability → 3 human-like classes
#    Low / Medium / High instead of just Low/High
# ─────────────────────────────────────────────
def predict_risk(case_data: dict) -> tuple:
    """
    Returns (risk_label, confidence_percentage).

    Probability thresholds:
      cancel_prob < 0.4  → Low
      cancel_prob < 0.7  → Medium
      cancel_prob >= 0.7 → High
    """
    df            = extract_ml_features(case_data)
    probabilities = rf_model.predict_proba(df)[0]

    # probabilities[1] = probability of cancellation (class 1)
    cancel_prob = float(probabilities[1])
    confidence  = round(max(probabilities) * 100, 1)

    # FIX 3: 3-class system based on cancellation probability
    if cancel_prob < 0.4:
        risk_label = "Low"
    elif cancel_prob < 0.7:
        risk_label = "Medium"
    else:
        risk_label = "High"

    return risk_label, confidence


# ─────────────────────────────────────────────
# 9. POST-RETRIEVAL FILTERING
#    FIX 2: filter out retrieved cases whose
#    lead_time is too far from current booking
#    so the LLM gets only logically similar cases
# ─────────────────────────────────────────────
def filter_docs(docs: list, case_data: dict) -> list:
    """
    Keep only retrieved cases where lead_time is within
    60 days of the current booking's lead_time.
    Falls back to all docs if filtering removes everything.
    """
    current_lead_time = case_data.get("lead_time", 0)

    filtered = [
        doc for doc in docs
        if abs((doc.metadata.get("lead_time") or 0) - current_lead_time) < 60
    ]

    # fallback: if filter is too strict, return original docs
    return filtered if filtered else docs


# ─────────────────────────────────────────────
# 10. FORMAT RETRIEVED DOCS
#     FIX 1: use categories instead of raw numbers
#     so LLM compares "long vs long" not "150 vs 136"
# ─────────────────────────────────────────────
def format_docs(docs: list) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata

        lead_time        = meta.get("lead_time") or 0
        adr              = meta.get("adr") or 0
        stay             = meta.get("total_stay") or 0
        special_requests = meta.get("total_of_special_requests") or 0
        prev_cancels     = meta.get("previous_cancellations") or 0

        # FIX 1: categories instead of raw numbers
        summary = (
            f"Case {i}: "
            f"Canceled={meta.get('is_canceled')}, "
            f"LeadTimeCategory={categorize_lead_time(lead_time)}, "
            f"Deposit={meta.get('deposit_type')}, "
            f"Market={meta.get('market_segment')}, "
            f"CustomerType={meta.get('customer_type')}, "
            f"PrevCancels={categorize_previous_cancellations(prev_cancels)}, "
            f"SpecialRequests={categorize_special_requests(special_requests)}, "
            f"Stay={categorize_total_stay(stay)}, "
            f"Rate={categorize_adr(adr)}"
        )
        lines.append(summary)

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 11. EXPLANATION PROMPT
#     FIX 4: stronger reasoning instructions
#     force LLM to compare categories explicitly
# ─────────────────────────────────────────────
explanation_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant for hotel booking cancellation risk analysis.

The risk level has already been determined by a trained machine learning model.
Your task is ONLY to explain that risk using the provided booking summary and similar past cases.

──────────────── RULES (STRICT) ────────────────
- DO NOT change or question the given risk level
- DO NOT use raw numbers (e.g., 100, 1, 0.5) anywhere in the explanation
- ONLY use category labels (e.g., "long lead time", "mid-range rate")
- ONLY use information explicitly provided in:
  1. Booking summary
  2. Similar past cases
- DO NOT assume or invent patterns (e.g., “high-risk channel”) unless clearly visible in similar cases
- If a feature is NOT present in similar cases, DO NOT use it in reasoning
- You MUST compare the current booking with similar cases
- Mention:
  • At least one similarity  
  • At least one difference  

──────────────── STYLE ────────────────
- Be concise and precise
- Use bullet points ONLY
- Do NOT mention case numbers (Case 1, Case 2, etc.)
- Refer to them as “similar past bookings”

──────────────── OUTPUT FORMAT ────────────────

Risk Level: {risk_level} (Model confidence: {confidence}%)

Reasoning:
- [Reason based ONLY on category labels]
- [Comparison with similar past bookings — include at least one similarity and one difference]

Recommendation:
- [One clear and actionable suggestion based on reasoning]

──────────────── INPUT DATA ────────────────

Booking summary:
{case_text}

Similar past cases:
{retrieved_context}
""")

output_parser = StrOutputParser()


# ─────────────────────────────────────────────
# 12. MAIN PIPELINE
# ─────────────────────────────────────────────
def analyze_booking(case_data: dict) -> dict:
    """
    Full RAG + ML pipeline:
    1. Convert features to categorical text for retrieval
    2. Retrieve similar past cases from Pinecone
    3. Filter retrieved cases by lead_time proximity
    4. Predict risk using Random Forest (3-class output)
    5. Generate explanation using Groq LLM (category-based)
    """

    # Step 1: Enriched categorical text for retrieval
    enriched_text = convert_features_to_text(case_data)

    # Step 2: Original case text for display
    case_text = input_to_case(case_data)

    # Step 3: Retrieve similar cases
    docs = retriever.invoke(enriched_text)

    # Step 4: Filter by lead_time proximity (FIX 2)
    filtered_docs = filter_docs(docs, case_data)

    # Step 5: Predict risk with 3-class system (FIX 3)
    risk_level, confidence = predict_risk(case_data)

    # Step 6: Handle no retrieved docs
    if not filtered_docs:
        return {
            "case_text": case_text,
            "enriched_text": enriched_text,
            "retrieved_count": 0,
            "risk_level": risk_level,
            "confidence": confidence,
            "analysis": (
                f"Risk Level: {risk_level} (Model confidence: {confidence}%)\n\n"
                f"Reasoning:\n"
                f"- No similar historical cases found in the database\n"
                f"- Risk was determined solely by the trained ML model\n\n"
                f"Recommendation:\n"
                f"- Review this booking manually before confirming"
            )
        }

    # Step 7: Format retrieved docs using categories (FIX 1)
    retrieved_context = format_docs(filtered_docs)

    # Step 8: Generate explanation with improved prompt (FIX 4)
    chain    = explanation_prompt | llm | output_parser
    analysis = chain.invoke({
        "risk_level": risk_level,
        "confidence": confidence,
        "case_text": case_text,
        "retrieved_context": retrieved_context
    })

    return {
        "case_text": case_text,
        "enriched_text": enriched_text,
        "retrieved_count": len(filtered_docs),
        "retrieved_cases": retrieved_context,
        "risk_level": risk_level,
        "confidence": confidence,
        "analysis": analysis
    }


# ─────────────────────────────────────────────
# 13. TEST CASES
# ─────────────────────────────────────────────
if __name__ == "__main__":

    case_A = {
        "hotel": "City Hotel",
        "lead_time": 150,
        "arrival_date_month": "August",
        "arrival_date_week_number": 32,
        "arrival_date_day_of_month": 10,
        "meal": "BB",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 2,
        "previous_bookings_not_canceled": 0,
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 90.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
        "room_mismatch": 1,
        "continent": "Europe",
        "total_guests": 2,
        "total_stay": 2
    }

    case_B = {
        "hotel": "Resort Hotel",
        "lead_time": 10,
        "arrival_date_month": "March",
        "arrival_date_week_number": 12,
        "arrival_date_day_of_month": 20,
        "meal": "HB",
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "is_repeated_guest": 1,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 2,
        "booking_changes": 1,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient-Party",
        "adr": 120.0,
        "required_car_parking_spaces": 1,
        "total_of_special_requests": 2,
        "room_mismatch": 0,
        "continent": "Europe",
        "total_guests": 2,
        "total_stay": 5
    }

    case_C = {
        "hotel": "City Hotel",
        "lead_time": 60,
        "arrival_date_month": "June",
        "arrival_date_week_number": 24,
        "arrival_date_day_of_month": 18,
        "meal": "BB",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 1,
        "previous_bookings_not_canceled": 1,
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 100.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 1,
        "room_mismatch": 0,
        "continent": "Europe",
        "total_guests": 2,
        "total_stay": 3
    }

    for name, case in zip(
        ["Case A (Risky)", "Case B (Safe)", "Case C (Mixed)"],
        [case_A, case_B, case_C]
    ):
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}\n")

        result = analyze_booking(case)

        print("Enriched embedding text:")
        print(result["enriched_text"])

        print(f"\nRetrieved cases : {result['retrieved_count']}")
        print(f"Predicted risk  : {result['risk_level']} ({result['confidence']}% confidence)\n")

        print("Analysis:")
        print(result["analysis"])
