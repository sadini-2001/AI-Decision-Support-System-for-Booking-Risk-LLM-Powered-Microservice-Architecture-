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
# 5. PINECONE CLIENT + EMBEDDINGS
#    We now use the Pinecone client directly
#    (not a LangChain retriever) so we can pass
#    a metadata filter at query time.
# ─────────────────────────────────────────────
pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("booking-decision-index")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Keep the LangChain vector store only for embedding convenience.
# We will NOT use its retriever — we call index.query() directly.
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# ─────────────────────────────────────────────
# 6. OUTCOME-ALIGNED RETRIEVAL
#
#  Strategy (3 tiers):
#
#  Tier 1 — PRIMARY (outcome-filtered):
#    Query Pinecone with a metadata filter so only
#    cases whose outcome matches the predicted risk
#    are returned.
#      High   → filter is_canceled == 1
#      Low    → filter is_canceled == 0
#      Medium → no outcome filter (mixed is fine)
#
#  Tier 2 — FALLBACK (unfiltered, if Tier 1 < 2 results):
#    Run an unfiltered query. This handles the "rare
#    case" scenario where the outcome class is thin
#    in the vector DB.
#
#  Tier 3 — MIXED CONTEXT (if both tiers fail):
#    Return an empty list so the pipeline gracefully
#    falls back to ML-only explanation.
#
#  Why 2 as the minimum threshold?
#    One case is not enough for a comparison bullet.
#    Two gives the LLM a similarity AND a difference.
# ─────────────────────────────────────────────
RETRIEVAL_K         = 5    # how many docs to request
RETRIEVAL_THRESHOLD = 0.65 # cosine similarity floor
FALLBACK_MIN        = 2    # min docs before triggering fallback


def _embed(text: str) -> list:
    """Return a raw embedding vector for a text string."""
    return embeddings.embed_query(text)


def _pinecone_query(
    vector: list,
    k: int,
    score_threshold: float,
    outcome_filter: Optional[dict] = None
) -> list:
    """
    Query Pinecone directly and return a list of dicts:
      [{"metadata": {...}, "score": float}, ...]

    outcome_filter example: {"is_canceled": {"$eq": 1}}
    """
    kwargs = {
        "vector": vector,
        "top_k": k,
        "include_metadata": True,
    }
    if outcome_filter:
        kwargs["filter"] = outcome_filter

    response = index.query(**kwargs)

    # Apply score threshold manually (Pinecone doesn't enforce it server-side)
    return [
        match for match in response["matches"]
        if match["score"] >= score_threshold
    ]


def retrieve_outcome_aligned(
    query_text: str,
    risk_level: str,
    case_data: dict
) -> list:
    """
    Outcome-aligned retrieval with lead-time post-filter.

    Returns a list of Pinecone match dicts that have passed:
      1. Cosine similarity threshold
      2. Outcome metadata filter (Tier 1) OR fallback (Tier 2)
      3. Lead-time proximity filter (±60 days)
    """
    query_vector      = _embed(query_text)
    current_lead_time = case_data.get("lead_time", 0)

    # ── Determine outcome filter based on predicted risk ──────────
    # High   → show canceled cases  (model says "this will cancel")
    # Low    → show not-canceled    (model says "this is safe")
    # Medium → no filter            (mixed context is appropriate)
    if risk_level == "High":
        outcome_filter = {"is_canceled": {"$eq": 1}}
    elif risk_level == "Low":
        outcome_filter = {"is_canceled": {"$eq": 0}}
    else:
        outcome_filter = None   # Medium: no filtering

    # ── Tier 1: outcome-filtered query ───────────────────────────
    matches = _pinecone_query(
        vector=query_vector,
        k=RETRIEVAL_K,
        score_threshold=RETRIEVAL_THRESHOLD,
        outcome_filter=outcome_filter,
    )

    # Apply lead-time proximity filter
    matches = _filter_by_lead_time(matches, current_lead_time)

    # ── Tier 2: fallback — unfiltered if Tier 1 insufficient ─────
    if len(matches) < FALLBACK_MIN and outcome_filter is not None:
        print(
            f"[WARN] Tier 1 returned only {len(matches)} result(s) "
            f"for risk={risk_level}. Falling back to unfiltered retrieval."
        )
        matches = _pinecone_query(
            vector=query_vector,
            k=RETRIEVAL_K,
            score_threshold=RETRIEVAL_THRESHOLD,
            outcome_filter=None,   # no outcome filter
        )
        matches = _filter_by_lead_time(matches, current_lead_time)

    return matches   # empty list → Tier 3 (handled in main pipeline)


def _filter_by_lead_time(matches: list, current_lead_time: int) -> list:
    """Keep only matches whose lead_time is within 60 days of current booking."""
    filtered = [
        m for m in matches
        if abs((m["metadata"].get("lead_time") or 0) - current_lead_time) < 60
    ]
    # If the filter is too strict, return original matches
    return filtered if filtered else matches


# ─────────────────────────────────────────────
# 7. GROQ LLM WRAPPER
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
# 8. CONVERT FEATURES TO EMBEDDING TEXT
#    (used for retrieval only — NOT shown to LLM)
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
# 9. BUILD CATEGORY-ONLY BOOKING SUMMARY FOR LLM
#    No raw numbers. Ordered by feature importance.
# ─────────────────────────────────────────────
def build_llm_booking_summary(case_data: dict) -> str:
    prev_cancels    = case_data.get("previous_cancellations", 0)
    prev_not_cancel = case_data.get("previous_bookings_not_canceled", 0)
    lead_time       = case_data.get("lead_time", 0)
    is_repeated     = case_data.get("is_repeated_guest", 0)
    room_mismatch   = case_data.get("room_mismatch", 0)
    deposit         = case_data.get("deposit_type", "Unknown")
    market          = case_data.get("market_segment", "Unknown")
    distribution    = case_data.get("distribution_channel", "Unknown")
    customer_type   = case_data.get("customer_type", "Unknown")
    hotel           = case_data.get("hotel", "Unknown")
    month           = case_data.get("arrival_date_month", "Unknown")
    meal            = case_data.get("meal", "Unknown")
    continent       = case_data.get("continent", "Unknown")

    lines = [
        f"- Cancellation history    : {categorize_previous_cancellations(prev_cancels)}",
        f"- Historical cancel rate  : {categorize_cancel_ratio(prev_cancels, prev_not_cancel)}",
        f"- Lead time               : {categorize_lead_time(lead_time)}",
        f"- Deposit type            : {deposit}",
        f"- Room assignment         : {'room type was changed at check-in' if room_mismatch else 'room assigned as booked'}",
        f"- Guest status            : {'returning guest' if is_repeated else 'new guest'}",
        f"- Market segment          : {market}",
        f"- Distribution channel    : {distribution}",
        f"- Customer type           : {customer_type}",
        f"- Daily rate category     : {categorize_adr(case_data.get('adr', 0))}",
        f"- Length of stay          : {categorize_total_stay(case_data.get('total_stay', 0))}",
        f"- Special requests        : {categorize_special_requests(case_data.get('total_of_special_requests', 0))}",
        f"- Booking changes         : {categorize_booking_changes(case_data.get('booking_changes', 0))}",
        f"- Waiting list            : {categorize_waiting_list(case_data.get('days_in_waiting_list', 0))}",
        f"- Hotel type              : {hotel}",
        f"- Arrival month           : {month}",
        f"- Meal plan               : {meal}",
        f"- Guest origin            : {continent}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 10. ML-BASED RISK PREDICTION
# ─────────────────────────────────────────────
def predict_risk(case_data: dict) -> tuple:
    df            = extract_ml_features(case_data)
    probabilities = rf_model.predict_proba(df)[0]
    cancel_prob   = float(probabilities[1])
    confidence    = round(max(probabilities) * 100, 1)

    if cancel_prob < 0.4:
        risk_label = "Low"
    elif cancel_prob < 0.7:
        risk_label = "Medium"
    else:
        risk_label = "High"

    return risk_label, confidence


# ─────────────────────────────────────────────
# 11. FORMAT RETRIEVED MATCHES — category labels only
#     Input is now a list of Pinecone match dicts,
#     not LangChain Document objects.
# ─────────────────────────────────────────────
def format_matches(matches: list) -> str:
    lines = []
    for i, match in enumerate(matches, start=1):
        meta = match["metadata"]

        lead_time        = meta.get("lead_time") or 0
        adr              = meta.get("adr") or 0
        stay             = meta.get("total_stay") or 0
        special_requests = meta.get("total_of_special_requests") or 0
        prev_cancels     = int(meta.get("previous_cancellations") or 0)
        prev_not_cancel  = int(meta.get("previous_bookings_not_canceled") or 0)
        is_canceled      = meta.get("is_canceled")

        # Normalize is_canceled — Pinecone may store as int or bool
        canceled_label = "Canceled" if is_canceled in (1, True, "1", "True") else "Not Canceled"

        summary = (
            f"Past booking {i}: "
            f"Outcome={canceled_label}, "
            f"Lead time={categorize_lead_time(lead_time)}, "
            f"Deposit={meta.get('deposit_type', 'Unknown')}, "
            f"Market={meta.get('market_segment', 'Unknown')}, "
            f"Customer type={meta.get('customer_type', 'Unknown')}, "
            f"Cancellation history={categorize_previous_cancellations(prev_cancels)}, "
            f"Historical cancel rate={categorize_cancel_ratio(prev_cancels, prev_not_cancel)}, "
            f"Room assignment={'changed' if meta.get('room_mismatch') else 'as booked'}, "
            f"Special requests={categorize_special_requests(special_requests)}, "
            f"Stay={categorize_total_stay(stay)}, "
            f"Rate={categorize_adr(adr)}"
        )
        lines.append(summary)

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 12. EXPLANATION PROMPT
# ─────────────────────────────────────────────
explanation_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant explaining hotel booking cancellation risk decisions.

The risk level was determined by a trained machine learning model. Your ONLY job is to
explain WHY using the booking summary and similar past bookings provided below.

══════════════════════════════════════════════
ABSOLUTE RULES — violating any rule is an error:
══════════════════════════════════════════════
1. DO NOT reproduce, reference, or infer any raw number (e.g. "110", "1", "0.5", "150 days").
   Use ONLY the category labels already present in the booking summary.
2. DO NOT change, question, or restate the risk level differently.
3. The booking summary below is the SINGLE source of truth for the current booking's features.
   DO NOT contradict it. If it says "one prior cancellation", use that — not zero, not two.
4. DO NOT invent patterns not visible in the similar past bookings.
5. You MUST prioritize HIGH-IMPACT features in your reasoning (listed below in order).
   Do NOT lead with low-impact features (e.g. special requests, meal plan).
6. You MUST compare the current booking with similar past bookings and state:
   • At least ONE clear similarity (a shared feature where past bookings had the same outcome)
   • At least ONE clear difference (where this booking differs from past ones)
7. Use bullet points only. No paragraphs. No case numbers (e.g. "Past booking 1").
   Refer to them collectively as "similar past bookings".
8. The similar past bookings have been pre-selected to match the predicted outcome.
   Use them as EVIDENCE supporting the risk decision — do not question the alignment.

══════════════════════════════════════════════
FEATURE IMPORTANCE ORDER (reason in this order):
══════════════════════════════════════════════
1. Cancellation history (prior cancellations + historical cancel rate)  ← HIGHEST IMPACT
2. Lead time
3. Deposit type
4. Room assignment (mismatch or not)
5. Guest status (new vs returning)
6. Market segment / distribution channel
7. Customer type
8. Daily rate category
9. Length of stay
10. Special requests / booking changes   ← LOWEST IMPACT — mention last or not at all

══════════════════════════════════════════════
OUTPUT FORMAT (follow exactly):
══════════════════════════════════════════════

Risk Level: {risk_level} (Model confidence: {confidence}%)

Reasoning:
- [Lead with the highest-impact features that support the risk level]
- [Next most important supporting feature]
- [Similarity with similar past bookings: what they share and how those bookings ended]
- [Difference from similar past bookings: what sets this booking apart]

Recommendation:
- [One clear, actionable suggestion tied directly to the reasoning above]

══════════════════════════════════════════════
INPUT DATA:
══════════════════════════════════════════════

Current booking (category labels only — use these as ground truth):
{llm_booking_summary}

Similar past bookings (pre-filtered to match predicted outcome):
{retrieved_context}
""")

output_parser = StrOutputParser()


# ─────────────────────────────────────────────
# 13. MAIN PIPELINE
# ─────────────────────────────────────────────
def analyze_booking(case_data: dict) -> dict:
    """
    Full RAG + ML pipeline with outcome-aligned retrieval:

    1.  Predict risk first (needed to determine retrieval filter)
    2.  Build categorical embedding text
    3.  Retrieve outcome-aligned similar cases from Pinecone
          Tier 1: metadata-filtered by predicted outcome
          Tier 2: unfiltered fallback if Tier 1 is too sparse
    4.  Build number-free LLM booking summary
    5.  Generate explanation using Groq LLM

    Key design decision — predict BEFORE retrieve:
    The retrieval filter depends on the predicted risk label,
    so prediction must come first. This is the correct order.
    """

    # Step 1: Predict risk FIRST — outcome-aligned retrieval needs this
    risk_level, confidence = predict_risk(case_data)

    # Step 2: Build categorical text for embedding
    enriched_text = convert_features_to_text(case_data)

    # Step 3: Human-readable case text for UI display (never passed to LLM)
    case_text = input_to_case(case_data)

    # Step 4: Outcome-aligned retrieval (Tier 1 → Tier 2 fallback)
    matches = retrieve_outcome_aligned(
        query_text=enriched_text,
        risk_level=risk_level,
        case_data=case_data,
    )

    # Step 5: Build number-free LLM booking summary
    llm_booking_summary = build_llm_booking_summary(case_data)

    # Step 6: Handle Tier 3 — no usable matches at all
    if not matches:
        return {
            "case_text": case_text,
            "enriched_text": enriched_text,
            "llm_booking_summary": llm_booking_summary,
            "retrieved_count": 0,
            "risk_level": risk_level,
            "confidence": confidence,
            "retrieval_mode": "none",
            "analysis": (
                f"Risk Level: {risk_level} (Model confidence: {confidence}%)\n\n"
                f"Reasoning:\n"
                f"- No similar historical cases found in the database\n"
                f"- Risk was determined solely by the trained ML model\n\n"
                f"Recommendation:\n"
                f"- Review this booking manually before confirming"
            )
        }

    # Step 7: Detect whether fallback was used (for logging/debugging)
    #         A match is "aligned" if its outcome matches predicted risk.
    expected_canceled_flag = 1 if risk_level == "High" else (0 if risk_level == "Low" else None)
    if expected_canceled_flag is not None:
        aligned_count = sum(
            1 for m in matches
            if m["metadata"].get("is_canceled") in (expected_canceled_flag, bool(expected_canceled_flag))
        )
        retrieval_mode = "outcome-aligned" if aligned_count >= FALLBACK_MIN else "fallback-unfiltered"
    else:
        retrieval_mode = "medium-unfiltered"   # Medium risk — intentionally mixed

    # Step 8: Format matches using category labels only
    retrieved_context = format_matches(matches)

    # Step 9: Generate explanation
    chain    = explanation_prompt | llm | output_parser
    analysis = chain.invoke({
        "risk_level": risk_level,
        "confidence": confidence,
        "llm_booking_summary": llm_booking_summary,
        "retrieved_context": retrieved_context,
    })

    return {
        "case_text": case_text,
        "enriched_text": enriched_text,
        "llm_booking_summary": llm_booking_summary,
        "retrieved_count": len(matches),
        "retrieved_cases": retrieved_context,
        "risk_level": risk_level,
        "confidence": confidence,
        "retrieval_mode": retrieval_mode,   # useful for monitoring
        "analysis": analysis,
    }


# ─────────────────────────────────────────────
# 14. TEST CASES
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
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}\n")

        result = analyze_booking(case)

        print("LLM Booking Summary (category-only):")
        print(result["llm_booking_summary"])

        print(f"\nRetrieved cases  : {result['retrieved_count']}")
        print(f"Retrieval mode   : {result['retrieval_mode']}")
        print(f"Predicted risk   : {result['risk_level']} ({result['confidence']}% confidence)\n")

        print("Analysis:")
        print(result["analysis"])
