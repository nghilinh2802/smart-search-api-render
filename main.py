from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import re
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import logging

app = FastAPI(title="Smart Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
df = None
product_embeddings = None
unique_keywords = set()

def init_firebase():
    if not firebase_admin._apps:
        firebase_creds = os.getenv('FIREBASE_CREDENTIALS')
        if firebase_creds:
            cred_dict = json.loads(firebase_creds)
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate("firebase-credentials.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

def init_model_and_data():
    global model, df, product_embeddings, unique_keywords
    try:
        logger.info("Loading SBERT model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        logger.info("Connecting to Firebase...")
        db = init_firebase()

        logger.info("Loading product data and embeddings...")
        df_loaded, embeddings = load_data_from_firestore(db)

        if df_loaded.empty or embeddings is None:
            raise RuntimeError("Product data or embeddings failed to load.")

        df = df_loaded
        product_embeddings = embeddings

        logger.info("Building keyword index...")
        setup_keywords()

        logger.info("✅ Model and data initialized.")

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

def load_data_from_firestore(db):
    products_ref = db.collection("products")
    docs = products_ref.stream()
    data = []

    for doc in docs:
        product = doc.to_dict()
        product["id"] = doc.id
        data.append(product)

    if not data:
        return pd.DataFrame(), None

    df = pd.DataFrame(data)
    df = preprocess_dataframe(df)

    embeddings_ref = db.collection("product_embeddings")
    embedding_docs = {doc.id: doc.to_dict() for doc in embeddings_ref.stream()}

    if len(embedding_docs) < len(df):
        raise RuntimeError("Not all products have saved embeddings in Firestore.")

    embeddings = []
    for _, row in df.iterrows():
        eid = row["id"]
        if eid in embedding_docs:
            embeddings.append(embedding_docs[eid]["embedding"])
        else:
            raise RuntimeError(f"Missing embedding for product ID: {eid}")

    return df, np.array(embeddings)

def preprocess_dataframe(df):
    def clean(text):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text).lower())).strip()

    df["clean_product_name"] = df["product_name"].fillna("").apply(clean)
    df["clean_description"] = df["description"].fillna("").apply(clean)
    df["clean_details"] = df["details"].fillna("").apply(clean)
    df["full_text"] = (df["clean_product_name"] + " " + df["clean_product_name"] +
                       " " + df["clean_description"] + " " + df["clean_details"])
    df["price_clean"] = df["price"].apply(lambda x: float(str(x).replace(",", "")) if pd.notna(x) else 0)
    df["parent_product_name"] = df["product_name"].fillna("").apply(lambda x: x.split('-')[0].split('(')[0].strip())
    return df

def setup_keywords():
    global unique_keywords
    for name in df["product_name"].dropna():
        clean_name = clean_text(name)
        unique_keywords.add(clean_name)
        for word in name.lower().split():
            unique_keywords.add(clean_text(word))
    extra = ["dog", "cat", "treat", "leash", "bed", "toy", "food"]
    for kw in extra:
        unique_keywords.add(clean_text(kw))

def clean_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text).lower())).strip()

def correct_spelling(word, keywords):
    word = word.lower()
    misspell = {"fodd": "food", "catnipp": "catnip", "collor": "collar", "leesh": "leash"}
    if word in misspell:
        return misspell[word]
    if word in keywords:
        return word
    match = get_close_matches(word, keywords, n=1, cutoff=0.6)
    return match[0] if match else word

def correct_full_query(query, keywords):
    return " ".join([correct_spelling(w, keywords) for w in query.split()])

def autocomplete_prefix(input_text, keywords, limit=5):
    input_text = input_text.lower()
    prefix = [kw for kw in keywords if kw.startswith(input_text)]
    partial = [kw for kw in keywords if input_text in kw and not kw.startswith(input_text)]
    return sorted(set(prefix + partial))[:limit]

def parse_price_range(query):
    q, minp, maxp = query, None, None
    r = re.search(r'price\s+under\s+([\d.,]+)', q, re.I)
    if r: maxp = float(r.group(1).replace(",", "")); q = re.sub(r'price\s+under\s+[\d.,]+', "", q, flags=re.I)
    r = re.search(r'price\s+over\s+([\d.,]+)', q, re.I)
    if r: minp = float(r.group(1).replace(",", "")); q = re.sub(r'price\s+over\s+[\d.,]+', "", q, flags=re.I)
    r = re.search(r'price\s+from\s+([\d.,]+)\s+to\s+([\d.,]+)', q, re.I)
    if r:
        minp = float(r.group(1).replace(",", ""))
        maxp = float(r.group(2).replace(",", ""))
        q = re.sub(r'price\s+from\s+[\d.,]+\s+to\s+[\d.,]+', "", q, flags=re.I)
    return q.strip(), minp, maxp

def search_sbert(query, top_n=10, price_min=None, price_max=None):
    if df is None or product_embeddings is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    q_clean = clean_text(query)
    q_vec = model.encode([q_clean])
    scores = cosine_similarity(q_vec, product_embeddings).flatten()
    df_copy = df.copy()
    df_copy["score"] = scores
    if price_min: df_copy = df_copy[df_copy["price_clean"] >= price_min]
    if price_max: df_copy = df_copy[df_copy["price_clean"] <= price_max]
    grouped = df_copy.groupby("parent_product_name").agg({"score": "max", "price_clean": "min", "id": "first"}).reset_index()
    grouped = grouped.sort_values("score", ascending=False).head(top_n).rename(columns={"price_clean": "price"})
    return grouped[["id", "parent_product_name", "price", "score"]]

class SearchRequest(BaseModel):
    query: str
    top_n: int = 10

class AutocompleteRequest(BaseModel):
    input_text: str
    limit: int = 5

@app.get("/")
async def root(): return {"status": "OK"}

@app.post("/search")
async def search_products(request: SearchRequest):
    q, minp, maxp = parse_price_range(request.query)
    q_corrected = correct_full_query(q, unique_keywords)
    results = search_sbert(q_corrected, request.top_n, minp, maxp)
    return {
        "corrected_query": q_corrected if q_corrected != q else None,
        "results": results.to_dict(orient="records")
    }

@app.post("/autocomplete")
async def autocomplete(request: AutocompleteRequest):
    word = request.input_text.lower()
    correct = correct_spelling(word, unique_keywords)
    suggestions = autocomplete_prefix(correct, unique_keywords, request.limit)
    return {
        "input": request.input_text,
        "correction": correct if correct != word else None,
        "suggestions": suggestions
    }

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Startup triggered")
    init_model_and_data()
