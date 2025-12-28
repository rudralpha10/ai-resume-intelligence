import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# match.py is inside: backend/matching/
# Go one level up → backend/
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RESUME_EMB_DIR = os.path.join(BASE_DIR, "data", "embeddings", "resumes")
JD_EMB_DIR = os.path.join(BASE_DIR, "data", "embeddings", "jobs")


def rank_resumes(jd_file, top_k=3):
    """
    Compare one job description embedding
    with all resume embeddings and rank them
    """

    jd_path = os.path.join(JD_EMB_DIR, jd_file)
    jd_embedding = np.load(jd_path)

    results = []

    for resume_file in os.listdir(RESUME_EMB_DIR):
        if resume_file.endswith(".npy"):
            resume_path = os.path.join(RESUME_EMB_DIR, resume_file)
            resume_embedding = np.load(resume_path)

            score = cosine_similarity(
                [jd_embedding],
                [resume_embedding]
            )[0][0]

            results.append(
                (resume_file.replace(".npy", ""), float(score))
            )

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]


# -------------------------------------------------
# Run directly
# -------------------------------------------------
if __name__ == "__main__":

    jd_file = "jd1.npy"  # change if needed
    matches = rank_resumes(jd_file)

    print(f"\n📄 Top resume matches for {jd_file}:\n")

    for idx, (resume, score) in enumerate(matches, start=1):
        print(f"{idx}. {resume}  →  Similarity: {score:.4f}")
