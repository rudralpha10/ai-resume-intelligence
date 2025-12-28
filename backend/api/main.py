import os
import uuid
import chromadb
import pdfplumber
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

# -------------------- APP --------------------
app = FastAPI(title="AI Resume Matcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import os
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf"
        os.environ["HF_HOME"] = "/tmp/hf"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        _model = SentenceTransformer(
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            device="cpu"
        )
    return _model



# -------------------- DB (CHROMA) --------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="resumes")

# -------------------- SCHEMA --------------------
class JobDescription(BaseModel):
    text: str
    top_k: int = 5

# -------------------- HELPERS --------------------
def extract_text_from_pdf(file: UploadFile) -> str:
    text = ""
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def safe_score(distance: float) -> float:
    if distance is None:
        return 0.0
    score = 1 - distance
    return round(max(0.0, min(score, 1.0)), 4)

# -------------------- ROUTES --------------------

@app.get("/")
def root():
    return {"status": "AI Resume Matcher API running"}

@app.get("/resumes")
def list_resumes():
    data = collection.get()
    return {
        "count": collection.count(),
        "resumes": data.get("ids", [])
    }

# -------- Upload ONE Resume --------
@app.post("/resume/upload")
async def upload_one(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file)
    model = get_model()
    embedding = model.encode(text).tolist()

    resume_id = f"{os.path.splitext(file.filename)[0]}-{uuid.uuid4().hex[:6]}"

    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[resume_id]
    )

    return {
        "message": "Resume uploaded successfully",
        "resume_id": resume_id
    }

# -------- Upload MULTIPLE Resumes --------
@app.post("/resumes/upload")
async def upload_multiple(files: List[UploadFile] = File(...)):
    model = get_model()
    uploaded = []

    for file in files:
        text = extract_text_from_pdf(file)
        embedding = model.encode(text).tolist()
        resume_id = f"{os.path.splitext(file.filename)[0]}-{uuid.uuid4().hex[:6]}"

        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[resume_id]
        )
        uploaded.append(resume_id)

    return {
        "message": "Resumes uploaded successfully",
        "uploaded": uploaded
    }

# -------- MATCH JOB DESCRIPTION --------
@app.post("/match")
def match_resumes(jd: JobDescription):
    if collection.count() == 0:
        return {"matches": []}

    model = get_model()
    jd_embedding = model.encode(jd.text).tolist()

    results = collection.query(
        query_embeddings=[jd_embedding],
        n_results=min(jd.top_k, collection.count())
    )

    matches = []
    for rid, dist in zip(results["ids"][0], results["distances"][0]):
        matches.append({
            "resume_id": rid,
            "score": safe_score(dist)
        })

    matches.sort(key=lambda x: x["score"], reverse=True)
    return {"matches": matches}
