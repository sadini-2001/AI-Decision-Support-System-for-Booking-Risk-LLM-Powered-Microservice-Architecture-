# 🏨 AI Booking Risk Decision Support System

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
Converted raw numerical values into semantic categories:
- `lead_time` → "long lead time"
- `adr` → "budget / premium rate"

This improved reasoning quality and similarity matching.

### 🔸 Model-driven Risk Thresholds
Used Random Forest probability outputs to define:
- **Low Risk** (< 0.4)
- **Medium Risk** (0.4 – 0.7)
- **High Risk** (> 0.7)

This avoided rigid rule-based thresholds.

### 🔸 Improved RAG Retrieval
- Applied similarity score threshold filtering
- Added domain-based filtering (e.g., lead time proximity)

Result: more relevant and realistic retrieved cases.

### 🔸 Prompt Engineering
- Enforced category-based comparisons
- Structured output (reasoning + recommendation)

This improved consistency and interpretability of LLM responses.

---

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|---|---|
| API connection issues | Proper port management & parallel service execution |
| Import/module errors | Restructured project with package-based imports |
| Noisy RAG retrieval | Added filtering + thresholds |
| Poor LLM explanations | Introduced feature categorization + prompt tuning |
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

### 🔹 3. Run Backend (FastAPI)

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Open in browser: http://127.0.0.1:8000/docs

### 🔹 4. Run Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

Open in browser: http://localhost:8501

---

## 📸 Demo
![Demo Screenshot](screenshots/demo_1.png)

- Rewrite the same thing without change anything as Readme.md file
