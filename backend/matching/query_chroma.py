import os
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma")

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(name="resumes")

print("📦 Total documents in collection:", collection.count())

def query_resumes(job_description: str, top_k: int = 3):
    jd_embedding = model.encode(job_description).tolist()

    results = collection.query(
        query_embeddings=[jd_embedding],
        n_results=top_k
    )

    return [
        {"resume_id": rid, "distance": float(dist)}
        for rid, dist in zip(results["ids"][0], results["distances"][0])
    ]

if __name__ == "__main__":
    jd = "Looking for ML engineer with Python and NLP experience"

    matches = query_resumes(jd)

    print("\n📄 Top matching resumes:\n")
    for i, m in enumerate(matches, 1):
        print(f"{i}. {m['resume_id']} → Distance: {m['distance']:.4f}")
