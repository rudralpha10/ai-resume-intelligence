import os
import chromadb
import pdfplumber
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

# -------------------- APP --------------------
app = FastAPI(title="AI Resume Matcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MODEL --------------------
model = SentenceTransformer("all-mpnet-base-v2")

# -------------------- DB --------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("resumes")

# -------------------- SCHEMA --------------------
class JobDescription(BaseModel):
    text: str
    top_k: int = 5

# -------------------- HELPERS --------------------
def extract_text_from_pdf(file: UploadFile):
    text = ""
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text.strip()

# -------------------- ROUTES --------------------
@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/resumes")
def list_resumes():
    return {
        "count": collection.count(),
        "ids": collection.get()["ids"]
    }

# -------- Upload ONE --------
@app.post("/resume/upload")
async def upload_one(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file)
    emb = model.encode(text).tolist()

    resume_id = file.filename.replace(".pdf", "")

    collection.add(
        documents=[text],
        embeddings=[emb],
        ids=[resume_id]
    )

    return {"uploaded": resume_id}

# -------- Upload MULTIPLE --------
@app.post("/resumes/upload")
async def upload_many(files: List[UploadFile] = File(...)):
    uploaded = []

    for file in files:
        text = extract_text_from_pdf(file)
        emb = model.encode(text).tolist()
        resume_id = file.filename.replace(".pdf", "")

        collection.add(
            documents=[text],
            embeddings=[emb],
            ids=[resume_id]
        )
        uploaded.append(resume_id)

    return {"uploaded": uploaded}

# -------- MATCH JD --------
@app.post("/match")
def match(jd: JobDescription):
    if collection.count() == 0:
        return {"matches": []}

    jd_emb = model.encode(jd.text).tolist()

    results = collection.query(
        query_embeddings=[jd_emb],
        n_results=min(jd.top_k, collection.count())
    )

    matches = []
    for rid, dist in zip(results["ids"][0], results["distances"][0]):
        score = round(1 - dist, 4) if dist is not None else 0.0
        matches.append({
            "resume_id": rid,
            "score": score
        })

    return {"matches": matches}
