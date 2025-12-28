import pdfplumber
import json
import sys
import re
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group() if match else None


def extract_phone(text):
    match = re.search(r"\b\d{10}\b", text)
    return match.group() if match else None


def extract_skills(text):
    skill_keywords = [
        "python", "machine learning", "deep learning", "nlp",
        "data analysis", "sql", "java", "c++", "tensorflow",
        "pandas", "numpy", "scikit-learn"
    ]
    text_lower = text.lower()
    return [skill for skill in skill_keywords if skill in text_lower]


def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None


def parse_resume(file_path):
    raw_text = extract_text_from_pdf(file_path)

    resume_json = {
        "name": extract_name(raw_text),
        "email": extract_email(raw_text),
        "phone": extract_phone(raw_text),
        "skills": extract_skills(raw_text),
        "education": None,
        "experience": None,
        "raw_text": raw_text
    }

    return resume_json


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    resume_path = sys.argv[1]
    output_path = sys.argv[2]

    parsed_data = parse_resume(resume_path)
    save_json(parsed_data, output_path)

    print("Resume parsed successfully!")
