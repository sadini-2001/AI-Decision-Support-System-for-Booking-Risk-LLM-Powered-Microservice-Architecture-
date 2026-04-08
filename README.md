# 🏨 AI-Powered Booking Risk Decision Support System

An end-to-end AI system that predicts hotel booking cancellation risk and provides human-like explanations using a hybrid approach combining Machine Learning and Retrieval-Augmented Generation (RAG).

---

## 🚀 Project Overview

Traditional machine learning models provide predictions but lack interpretability. This project goes beyond prediction by building a decision support system that answers:

- ❓ Will a booking be canceled?
- ❓ Why is it risky?
- ❓ What similar past cases support this?
- ❓ What action should be taken?

---

## 🧠 System Architecture

```
Frontend (Streamlit)
        ↓
Backend API (FastAPI)
        ↓
ML Model (Random Forest) + RAG Pipeline (LangChain)
        ↓
Vector DB (Pinecone) + LLM (Groq)
```

### Internal Pipeline

```
Input booking dict
        │
        ▼
┌───────────────────┐
│  predict_risk()   │  ← Random Forest (sklearn)
│  3-class output   │    Low / Medium / High
└────────┬──────────┘
         │  risk_level needed BEFORE retrieval
         ▼
┌──────────────────────────────┐
│ retrieve_outcome_aligned()   │  ← Pinecone (direct query)
│  Tier 1: outcome-filtered    │    High  → is_canceled == 1
│  Tier 2: unfiltered fallback │    Low   → is_canceled == 0
│  Tier 3: empty → ML-only     │    Medium → no filter
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ build_llm_booking_summary│  ← Category labels only, zero raw numbers
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  explanation_prompt      │  ← Groq LLaMA 3.1 8B
│  + format_matches()      │    Feature-importance-ordered reasoning
└────────┬─────────────────┘
         │
         ▼
    analysis dict
```

---

## 🔍 Features

- 📊 Booking cancellation prediction (Random Forest)
- 🔎 Similar case retrieval using vector search (RAG)
- 🧠 AI-generated explanations using LLM
- 🎯 Risk classification (Low / Medium / High)
- 💬 Structured reasoning + recommendation
- 🌐 Interactive UI for real-time analysis

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** – API development
- **Pydantic** – Data validation

### Frontend
- **Streamlit** – Interactive UI

### Machine Learning
- **Scikit-learn** (Random Forest)

### AI / GenAI
- **LangChain** – RAG pipeline orchestration
- **Hugging Face** (sentence-transformers) – Embeddings
- **Pinecone** – Vector database
- **Groq LLM** – Explanation generation

---

## 💡 Key Design Decisions

### 🔸 Feature Categorization

Converted raw numerical values into semantic categories before passing anything to the LLM:

| Feature | Raw value | Category label |
|---|---|---|
| `lead_time` | `150` | `long lead time` |
| `adr` | `92.5` | `budget rate` |
| `total_stay` | `2` | `short stay` |
| `previous_cancellations` | `1` | `one prior cancellation` |
| `total_of_special_requests` | `3` | `several special requests` |
| `cancel_ratio` | `0.67` | `moderate historical cancellation rate` |

This improved reasoning quality and similarity matching, and eliminated a class of LLM errors where raw numbers were leaked or misinterpreted.

### 🔸 Model-Driven Risk Thresholds

Used Random Forest probability outputs to define three classes:

| Label | Cancellation probability | Meaning |
|---|---|---|
| `Low` | < 40% | Booking is likely to be honoured |
| `Medium` | 40% – 70% | Uncertain — monitor closely |
| `High` | ≥ 70% | Booking is likely to cancel |

This avoided rigid rule-based thresholds.

### 🔸 Handling Evidence Conflict in RAG (Key Innovation)

In similarity-based retrieval, bookings with similar features can have different outcomes (some canceled, some not). This created a critical issue:

- Retrieved cases sometimes contradicted the model's prediction
- Leading to inconsistent and confusing explanations

```
Standard retrieval (the problem):

Booking features  →  retrieve top-5 similar cases
                           │
                    ┌──────┴──────┐
                    │             │
               Canceled      Not Canceled
               (2 cases)     (3 cases)
                    │             │
                    └──────┬──────┘
                           │
                    LLM sees mixed evidence
                           │
                    "Model says High Risk,
                     but 3 similar bookings
                     did NOT cancel..." ← CONTRADICTION
```

**👉 Solution: Outcome-Conditional Retrieval**

Retrieval is conditioned on the predicted outcome so the LLM always receives evidence consistent with the model's decision:

```
predict_risk()  ──►  "High"
                        │
                        ▼
              Apply Pinecone metadata filter:
              { "is_canceled": { "$eq": 1 } }
                        │
                        ▼
              Top-5 similar cases, ALL canceled
                        │
                        ▼
              LLM receives coherent evidence:
              "Similar bookings with these features
               DID cancel — here is why this one
               is also High Risk."
```

The key architectural insight is that **prediction must happen before retrieval** — the retrieved evidence is selected to support the model's conclusion, not contradict it.

**The 3-Tier Retrieval Strategy:**

```
┌─────────────────────────────────────────────────────────────┐
│  TIER 1 — Outcome-Filtered Query  (primary)                 │
│                                                             │
│  High Risk  →  filter: is_canceled == 1                     │
│  Low Risk   →  filter: is_canceled == 0                     │
│  Medium     →  no filter  (mixed context is appropriate)    │
│                                                             │
│  Requires: ≥ 2 results above cosine threshold (0.65)        │
└──────────────────────┬──────────────────────────────────────┘
                       │ < 2 results? → escalate
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  TIER 2 — Unfiltered Fallback                               │
│                                                             │
│  Drops the outcome filter. Returns the closest neighbors    │
│  regardless of cancellation outcome.                        │
│  Triggered when canceled cases are rare in the DB.          │
└──────────────────────┬──────────────────────────────────────┘
                       │ still 0 results? → escalate
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  TIER 3 — ML-Only Explanation                               │
│                                                             │
│  No retrieved cases at all. Returns a graceful fallback:    │
│  "Risk determined solely by the trained ML model.           │
│   Review this booking manually."                            │
└─────────────────────────────────────────────────────────────┘
```

**Result:**
- ✅ More reliable explanations
- ✅ Better alignment between ML predictions and LLM reasoning
- ✅ Improved trust in the system

### 🔸 Improved RAG Retrieval

- Applied similarity score threshold filtering (cosine ≥ 0.65)
- Added lead-time proximity filtering (±60 days)
- Switched from LangChain `.as_retriever()` to direct `index.query()` calls to enable server-side metadata filtering

Result: more relevant and realistic retrieved cases.

### 🔸 Prompt Engineering

- Enforced category-based comparisons (no raw numbers allowed)
- Defined explicit feature importance order for reasoning (cancellation history → lead time → deposit → … → special requests)
- Structured output format (reasoning bullets + one recommendation)
- LLM required to state at least one similarity AND one difference vs. past bookings

This improved consistency and interpretability of LLM responses.

---

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|---|---|
| API connection issues | Proper port management & parallel service execution |
| Noisy RAG retrieval | Added filtering + score thresholds |
| Poor LLM explanations | Introduced feature categorization + prompt tuning |
| Evidence conflict in RAG | Outcome-conditional retrieval (Tier 1 → 2 → 3) |
| Raw numbers leaking to LLM | Separated `build_llm_booking_summary()` from UI display text |
| System instability | Debugged components independently |

---

## ▶️ How to Run the Project

### 🔹 1. Clone the repository

```bash
git clone <your-repo-link>
cd AI_DecisionSupportSystem_for_BookingRisk
```

### 🔹 2. Activate virtual environment

```bash
.venv\Scripts\Activate
```

### 🔹 3. Configure environment variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

### 🔹 4. Run Backend (FastAPI)

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Open in browser: http://127.0.0.1:8000/docs

### 🔹 5. Run Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

Open in browser: http://localhost:8501

---

### Retrieval Mode Reference

| Value | Meaning |
|---|---|
| `outcome-aligned` | Tier 1 succeeded — retrieved cases match predicted outcome |
| `fallback-unfiltered` | Tier 2 triggered — not enough outcome-matching cases in DB |
| `medium-unfiltered` | Medium risk — intentionally mixed retrieval |
| `none` | Tier 3 — no usable matches, ML-only explanation returned |

---

## 📸 Demo
![Demo Screenshot](screenshots/demo_1.png)
![Demo Screenshot](screenshots/demo_2.png)
![Demo Screenshot](screenshots/demo_3.png)
![Demo Screenshot](screenshots/demo_4.png)




