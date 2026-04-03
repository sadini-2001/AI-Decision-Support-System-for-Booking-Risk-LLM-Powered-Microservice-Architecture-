import os
import pandas as pd
from tqdm import tqdm  # progress bar
from dotenv import load_dotenv  # load .env
from sentence_transformers import SentenceTransformer  # embedding model
from pinecone import Pinecone, ServerlessSpec  # vector DB
from huggingface_hub import login  # HF login

# ================================
# Load environment variables
# ================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# login to HF (optional)
if HF_TOKEN:
    login(token=HF_TOKEN)

# ================================
# Load dataset
# ================================
df = pd.read_csv("../Data/cases.csv")

# ================================
# Load embedding model
# ================================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ================================
# Initialize Pinecone
# ================================
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "booking-decision-index"

# ================================
# DELETE old index (fix old wrong data)
# ================================
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)  # remove old data

# ================================
# Create new index
# ================================
pc.create_index(
    name=index_name,
    dimension=384,  # embedding size
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# connect to index
index = pc.Index(index_name)

# ================================
# Prepare + Upload (memory safe)
# ================================

batch_size = 50  # smaller batch
batch = []       # temporary storage

for i, row in tqdm(df.iterrows(), total=len(df)):

    text = row["case_text"]  # case text

    # create embedding
    embedding = model.encode(text).tolist()

    # metadata with text (important fix)
    metadata = {
        "text": text,  # required for LangChain
        "case_id": int(row["case_id"]),
        "is_canceled": int(row["is_canceled"]),
        "lead_time": float(row["lead_time"]),
        "deposit_type": str(row["deposit_type"]),
        "market_segment": str(row["market_segment"]),
        "customer_type": str(row["customer_type"]),
        "previous_cancellations": float(row["previous_cancellations"]),
        "booking_changes": float(row["booking_changes"]),
        "total_of_special_requests": float(row["total_of_special_requests"]),
        "adr": float(row["adr"]),
        "total_guests": float(row["total_guests"]),
        "total_stay": float(row["total_stay"]),
        "continent": str(row["continent"]),
        "room_mismatch": int(row["room_mismatch"])
    }

    # add to batch
    batch.append((
        str(row["case_id"]),
        embedding,
        metadata
    ))

    # upload when batch is full
    if len(batch) == batch_size:
        index.upsert(batch)
        batch = []  # clear memory

# upload remaining data
if batch:
    index.upsert(batch)

print("All data uploaded to Pinecone!")

# ================================
# Quick test
# ================================
query = "Long lead time with no deposit booking"
query_embedding = model.encode(query).tolist()

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

print("\nSimilar cases:\n")

for match in results["matches"]:
    print(match["metadata"])
