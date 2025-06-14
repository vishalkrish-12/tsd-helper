import os
import chromadb
import requests # Import the requests library
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables from .env file (optional)
# load_dotenv()

# --- API and Proxy Configuration ---
# API_PROXY_URL = "https://aipipe.org/v1/chat/completions"
API_PROXY_URL =  "https://aipipe.org/openrouter/v1/chat/completions"
API_KEY = os.getenv("AIPIPE_TOKEN")
if not API_KEY:
    raise Exception("The AIPIPE_TOKEN environment variable is not set. Please set it before running the app.")

# --- Initialize ChromaDB Client ---
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "tds_virtual_ta"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# --- Pydantic Models for API Data Structure ---
class StudentRequest(BaseModel):
    question: str
    image: Optional[str] = Field(None, description="Optional base64 encoded image string.")

class Link(BaseModel):
    url: str
    text: str

class ApiResponse(BaseModel):
    answer: str
    links: List[Link]

# --- FastAPI Application ---
app = FastAPI(
    title="TDS Virtual TA (AIProxy Version)",
    description="An API for answering student questions using the aipipe.org proxy.",
    version="1.1.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Initialize embedding model globally (so it's loaded only once)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Core RAG Logic ---
def query_rag(question: str, image: str = None) -> ApiResponse:
    # 1. Retrieval: Find relevant context
    results = collection.query(query_texts=[question], n_results=30, include=['documents', 'metadatas', 'embeddings'])
        # Defensive check for empty results
    if (
        not results or
        not results.get('documents') or not results['documents'] or not results['documents'][0]
    ):
        return ApiResponse(
            answer="Sorry, I couldn't find any relevant information in the knowledge base.",
            links=[]
        )

    retrieved_documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    doc_embeddings = results['embeddings'][0]

    # 2. Embed the user's question
    question_embedding = embedding_model.encode([question])[0]

    # 3. Re-rank by cosine similarity
    doc_scores = [
        (cosine_similarity(question_embedding, doc_emb), doc, meta)
        for doc_emb, doc, meta in zip(doc_embeddings, retrieved_documents, metadatas)
    ]
    doc_scores.sort(reverse=True, key=lambda x: x[0])
    top_n = 10
    top_docs = doc_scores[:top_n]
    context = "\n\n---\n\n".join([doc for _, doc, _ in top_docs])
    top_metadatas = [meta for _, _, meta in top_docs]
    

    # 2. Augmentation: Construct the prompt
    prompt_template = f""" You are a helpful teaching assistant. Use the following context to answer the student's question
    CONTEXT:
    {context}

    STUDENT'S QUESTION:
    {question}
    """

    # If image is provided, add a note to the prompt
    if image:
        prompt_template += "\n\nNOTE: The student attached an image, but image understanding is not supported."

    messages = [
        {"role": "system", "content": "You are a helpful teaching assistant."},
        {"role": "user", "content": prompt_template}
    ]

    # 3. Generation: Call the AIProxy endpoint
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4.1-nano",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 200
    }

    try:
        response = requests.post(API_PROXY_URL, headers=headers, json=payload)
        response.raise_for_status()
        api_response_json = response.json()
        answer = api_response_json["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling AIProxy: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to get a response from the AI proxy: {e}")
    except (KeyError, IndexError) as e:
        print(f"Error parsing AIProxy response: {e}")
        raise HTTPException(status_code=500, detail="Invalid response format from the AI proxy.")

    # Format links from the retrieved documents' metadata
    # links = [
    #     Link(url=meta['source'], text=doc)
    #     for meta, doc in zip(metadatas, retrieved_documents)
    #     if meta.get('source', '').startswith('http')
    # ]
    links = [
        Link(url=meta['source'], text=doc)
        for (_, doc, meta) in top_docs
        if meta.get('source', '').startswith('https://') or meta.get('source', '').startswith('http://')
    ]

    return ApiResponse(answer=answer, links=links)

@app.post("/api/", response_model=ApiResponse)
async def handle_question(request: StudentRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question field cannot be empty.")

    return query_rag(request.question, request.image)

@app.get("/", include_in_schema=False)
def root():
    return {"message": "TDS Virtual TA API is running. Send POST requests to /api/"}