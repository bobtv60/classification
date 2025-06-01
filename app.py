from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import wordninja
from better_profanity import profanity

app = FastAPI()

# Load the profanity filter word list
profanity.load_censor_words()

# Zero-shot classifier (you can replace or expand this)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Candidate labels for classification
LABELS = ["bug", "suggestion", "spam", "rude", "other"]
CONFIDENCE_THRESHOLD = 0.5

class Feedback(BaseModel):
    text: str

def contains_profanity_wordninja(text: str) -> bool:
    # Split concatenated words into likely words
    words = wordninja.split(text.lower())
    # Check each word for profanity
    for word in words:
        if profanity.contains_profanity(word):
            return True
    return False

@app.post("/classify")
async def classify(feedback: Feedback):
    # Check profanity first
    if contains_profanity_wordninja(feedback.text):
        return {"label": "rude", "score": 1.0}

    # Run zero-shot classification on the original text
    result = classifier(feedback.text, candidate_labels=LABELS)
    best_label = result["labels"][0]
    best_score = result["scores"][0]

    if best_score < CONFIDENCE_THRESHOLD:
        return {"label": "other", "score": best_score}

    return {"label": best_label, "score": best_score}
