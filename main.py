# ==================== main.py ====================
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
from typing import Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Search API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
df = None
product_embeddings = None
unique_keywords = set()

# Initialize Firebase
def init_firebase():
    try:
        # Check if Firebase is already initialized
        if not firebase_admin._apps:
            # Get Firebase credentials from environment variable
            firebase_creds = os.getenv('FIREBASE_CREDENTIALS')
            if firebase_creds:
                # Parse JSON from environment variable
                cred_dict = json.loads(firebase_creds)
                cred = credentials.Certificate(cred_dict)
            else:
                # Fallback to service account file
                cred = credentials.Certificate("firebase-credentials.json")
            
            firebase_admin.initialize_app(cred)
        
        return firestore.client()
    except Exception as e:
        logger.error(f"Firebase initialization error: {e}")
        raise

# Initialize model and data
def init_model_and_data():
    global model, df, product_embeddings, unique_keywords
    
    try:
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Loading data from Firestore...")
        db = init_firebase()
        df, product_embeddings = load_data_from_firestore(db)
        
        logger.info("Setting up keywords...")
        setup_keywords()
        
        logger.info("Initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

def load_data_from_firestore(db):
    try:
        # Load products
        products_ref = db.collection("products")
        docs = products_ref.stream()
        data = []
        
        for doc in docs:
            product = doc.to_dict()
            product['id'] = doc.id
            data.append(product)
        
        if not data:
            logger.warning("No products found in Firestore")
            return pd.DataFrame(), np.array([])
        
        df = pd.DataFrame(data)
        
        # Preprocess data
        df = preprocess_dataframe(df)
        
        # Try to load embeddings from Firestore first
        try:
            embeddings_ref = db.collection("product_embeddings")
            embedding_docs = {doc.id: doc.to_dict() for doc in embeddings_ref.stream()}
            
            if len(embedding_docs) == len(df):
                logger.info("Loading existing embeddings from Firestore...")
                embeddings = []
                for _, row in df.iterrows():
                    if row['id'] in embedding_docs:
                        embeddings.append(embedding_docs[row['id']]['embedding'])
                    else:
                        raise Exception("Missing embedding")
                
                product_embeddings = np.array(embeddings)
            else:
                raise Exception("Embedding count mismatch")
                
        except Exception as e:
            logger.info(f"Generating new embeddings: {e}")
            # Generate new embeddings
            product_texts = df["full_text"].tolist()
            product_embeddings = model.encode(product_texts, show_progress_bar=False)
            
            # Save embeddings to Firestore
            try:
                for idx, embedding in enumerate(product_embeddings):
                    db.collection("product_embeddings").document(df.iloc[idx]["id"]).set({
                        "product_id": df.iloc[idx]["id"],
                        "embedding": embedding.tolist()
                    })
                logger.info("Embeddings saved to Firestore")
            except Exception as save_error:
                logger.warning(f"Failed to save embeddings: {save_error}")
        
        return df, product_embeddings
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_dataframe(df):
    # Clean text function
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Preprocess columns
    df["clean_product_name"] = df["product_name"].fillna("").apply(clean_text)
    df["clean_description"] = df["description"].fillna("").apply(clean_text)
    df["clean_details"] = df["details"].fillna("").apply(clean_text)
    
    # Weight product_name higher by repeating it
    df["full_text"] = (df["clean_product_name"] + " " + df["clean_product_name"] + " " +
                       df["clean_description"] + " " + df["clean_details"])

    # Clean price
    def clean_price(price):
        if pd.isna(price):
            return 0
        try:
            return float(str(price).replace(",", "").strip())
        except:
            return 0
    
    df["price_clean"] = df["price"].apply(clean_price)

    # Extract parent product name
    def extract_parent_name(name):
        if pd.isna(name):
            return ""
        if '-' in name:
            return name.split('-')[0].strip()
        elif '(' in name:
            return name.split('(')[0].strip()
        else:
            return name.strip()
    
    df["parent_product_name"] = df["product_name"].apply(extract_parent_name)
    
    return df

def setup_keywords():
    global unique_keywords
    
    # Add product names and words
    for name in df["product_name"].dropna().unique():
        clean_name = clean_text(name)
        unique_keywords.add(clean_name)
        for word in name.lower().split():
            clean_word = clean_text(word)
            if clean_word:
                unique_keywords.add(clean_word)
    
    # Add categories and brands
    categories_and_brands = [
        'Accessories', 'Apparel & Costume', 'Bed', 'Blanket', 'Brushes & Combs', 'Carriers & Kennels',
        'Cat', 'Collar', 'Collar & Leash', 'Costume', 'Dental care', 'Deodorant tools', 'Dog', 'Dry food',
        'Feeders', 'Flea and Tick control', 'Food', 'Hammock', 'Leash', 'Nail care', 'Pillow', 'Set',
        'Shampoo & Conditioner', 'Small Animal', 'Supplements & Vitamins', 'Toys', 'Training', 'Treat',
        'Wet food', 'catnip', '3 Peaks', 'BAM!', 'Barkbutler', 'Basil', 'Chuckit!', 'Coachi', 'M-PETS',
        'Mitag', 'Noble', 'PAW', 'Papa Pawsome', 'Papaw Cartel', 'Pawgypets', 'Pawsome Couture',
        'Pedigree', 'Pets at Home', 'QPets', 'Squeeezys', 'TOPDOG', 'Trixie', 'dog food', 'cat food'
    ]
    
    for kw in categories_and_brands:
        clean_kw = clean_text(kw)
        unique_keywords.add(clean_kw)
        for part in kw.split():
            clean_part = clean_text(part)
            if clean_part:
                unique_keywords.add(clean_part)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Common misspellings dictionary
misspellings = {
    "fodd": "food",
    "catnipp": "catnip",
    "collor": "collar",
    "leesh": "leash"
}

def correct_spelling(word, keyword_list):
    word_lower = word.lower()
    if word_lower in misspellings:
        return misspellings[word_lower]
    if word_lower in keyword_list:
        return word_lower
    matches = get_close_matches(word_lower, keyword_list, n=1, cutoff=0.6)
    return matches[0] if matches else word_lower

def correct_full_query(query, keyword_list):
    words = query.lower().split()
    corrected_words = [correct_spelling(w, keyword_list) for w in words]
    return " ".join(corrected_words)

def autocomplete_prefix(input_text, keyword_list, limit=5):
    input_text = input_text.lower()
    suggestions = []
    # Exact prefix matches
    suggestions.extend(kw for kw in keyword_list if kw.startswith(input_text))
    # Partial matches within words
    suggestions.extend(kw for kw in keyword_list if input_text in kw and not kw.startswith(input_text))
    return sorted(set(suggestions))[:limit]

def parse_price_range(query):
    price_min = None
    price_max = None
    
    # Price under
    m = re.search(r'price\s+under\s+([\d.,]+)', query, re.IGNORECASE)
    if m:
        price_max = float(re.sub(r"[.,]", "", m.group(1)))
        query = re.sub(r'price\s+under\s+[\d.,]+', '', query, flags=re.IGNORECASE).strip()
        return query, price_min, price_max
    
    # Price over
    m = re.search(r'price\s+over\s+([\d.,]+)', query, re.IGNORECASE)
    if m:
        price_min = float(re.sub(r"[.,]", "", m.group(1)))
        query = re.sub(r'price\s+over\s+[\d.,]+', '', query, flags=re.IGNORECASE).strip()
        return query, price_min, price_max
    
    # Price range
    m = re.search(r'price\s+from\s+([\d.,]+)\s+to\s+([\d.,]+)', query, re.IGNORECASE)
    if m:
        price_min = float(re.sub(r"[.,]", "", m.group(1)))
        price_max = float(re.sub(r"[.,]", "", m.group(2)))
        query = re.sub(r'price\s+from\s+[\d.,]+\s+to\s+[\d.,]+', '', query, flags=re.IGNORECASE).strip()
        return query, price_min, price_max
    
    return query.strip(), price_min, price_max

def search_sbert(query, top_n=10, price_min=None, price_max=None):
    if df is None or product_embeddings is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    clean_query = clean_text(query)
    query_embedding = model.encode([clean_query])
    scores = cosine_similarity(query_embedding, product_embeddings).flatten()
    
    df_search = df.copy()
    df_search["score"] = scores
    
    # Apply price filters
    if price_min is not None:
        df_search = df_search[df_search["price_clean"] >= price_min]
    if price_max is not None:
        df_search = df_search[df_search["price_clean"] <= price_max]
    
    # Group by parent product name
    grouped = df_search.groupby("parent_product_name").agg({
        "score": "max",
        "price_clean": "min",
        "id": "first"
    }).reset_index()
    
    grouped = grouped.sort_values(by="score", ascending=False).head(top_n)
    grouped = grouped.rename(columns={"price_clean": "price"})
    
    return grouped[["id", "parent_product_name", "price", "score"]]

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    top_n: int = 10

class AutocompleteRequest(BaseModel):
    input_text: str
    limit: int = 5

class SimilarItemsRequest(BaseModel):
    product_id: str
    top_n: int = 5

# API Routes
@app.get("/")
async def root():
    return {"message": "Smart Search API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": df is not None,
        "products_count": len(df) if df is not None else 0
    }

@app.post("/search")
async def search_products(request: SearchRequest):
    try:
        if df is None or product_embeddings is None:
            raise HTTPException(status_code=500, detail="Service not initialized")
        
        query, price_min, price_max = parse_price_range(request.query)
        corrected_query = correct_full_query(query, unique_keywords)
        results = search_sbert(corrected_query, request.top_n, price_min, price_max)
        
        return {
            "corrected_query": corrected_query if corrected_query != query else None,
            "results": results.to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/autocomplete")
async def autocomplete(request: AutocompleteRequest):
    try:
        if not unique_keywords:
            raise HTTPException(status_code=500, detail="Keywords not initialized")
        
        input_text = request.input_text.lower()
        corrected = correct_spelling(input_text, list(unique_keywords))
        
        if corrected and corrected != input_text:
            fallback = autocomplete_prefix(corrected, unique_keywords, request.limit)
            return {
                "type": "spell_corrected",
                "input": input_text,
                "suggestions": fallback,
                "correction": corrected
            }
        
        suggestions = autocomplete_prefix(input_text, unique_keywords, request.limit)
        if suggestions:
            return {
                "type": "autocomplete",
                "input": input_text,
                "suggestions": suggestions,
                "correction": None
            }
        
        return {
            "type": "no_match",
            "input": input_text,
            "suggestions": [],
            "correction": None
        }
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similar_items")
async def get_similar_items(request: SimilarItemsRequest):
    try:
        if df is None or product_embeddings is None:
            raise HTTPException(status_code=500, detail="Service not initialized")
        
        if request.product_id not in df['id'].values:
            raise HTTPException(status_code=404, detail="Product ID not found")
        
        # Get the embedding of the target product
        target_idx = df[df['id'] == request.product_id].index[0]
        target_embedding = product_embeddings[target_idx].reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(target_embedding, product_embeddings).flatten()
        
        # Get top similar products (excluding the target product itself)
        top_indices = np.argpartition(similarities, -(request.top_n + 1))[-(request.top_n + 1):]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        top_indices = [i for i in top_indices if df['id'].iloc[i] != request.product_id][:request.top_n]
        
        # Return similar products
        similar_products = df.iloc[top_indices][['id', 'product_name', 'price_clean']].copy()
        similar_products['score'] = similarities[top_indices]
        similar_products = similar_products.rename(columns={'price_clean': 'price'})
        
        return similar_products.to_dict(orient='records')
        
    except Exception as e:
        logger.error(f"Similar items error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Smart Search API...")
    init_model_and_data()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)