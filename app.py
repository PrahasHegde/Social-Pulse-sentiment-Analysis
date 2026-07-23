import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess import SocialMediaPreprocessor

app = FastAPI(
    title="Next-Gen Sentiment & Intelligence API",
    version="2.0.0",
    description="Microservice providing Sentiment, Aspect Extraction, and Emotion Analysis."
)

MODEL_DIR = "./model_assets"
preprocessor = SocialMediaPreprocessor()

# Load Model Assets
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
except Exception as e:
    # Fallback to remote model if local model assets are missing
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

# Official Label Mapping for twitter-roberta-base-sentiment-latest (0: Negative, 1: Neutral, 2: Positive)
LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}

class AnalysisRequest(BaseModel):
    text: str = Field(..., example="I love the new app interface, but it keeps crashing when opening settings!")

class AnalysisResponse(BaseModel):
    sentiment_label: str
    confidence: float
    probabilities: dict
    aspects: list
    emotions: dict

def simulate_emotion_intensity(scores: dict, text: str) -> dict:
    """Calculates multidimensional emotional intensity based on sentiment probabilities and text cues."""
    neg, neu, pos = scores["negative"], scores["neutral"], scores["positive"]
    
    return {
        "Joy": round(pos * 0.9, 2),
        "Anger": round(neg * 0.7 if "!" in text or "bad" in text.lower() else neg * 0.4, 2),
        "Sadness": round(neg * 0.5, 2),
        "Surprise": round(0.3 if "!" in text else 0.1, 2),
        "Disgust": round(neg * 0.6 if "crash" in text.lower() or "hate" in text.lower() else neg * 0.2, 2),
        "Trust": round(pos * 0.8 + neu * 0.2, 2)
    }

@app.get("/")
async def root():
    return {"status": "online", "message": "API Active. Access /docs for interactive documentation."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    try:
        # 1. Cleaning
        cleaned_text = preprocessor.clean_text(request.text)

        # 2. Tokenize & Predict Sentiment
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()

        predicted_idx = int(np.argmax(probs))
        
        # Format scores with explicit label mappings
        scores = {LABEL_MAPPING[i]: float(probs[i]) for i in range(len(probs))}
        
        # 3. Extract Aspects & Emotions
        aspects = preprocessor.extract_aspects(request.text)
        emotions = simulate_emotion_intensity(scores, request.text)

        return AnalysisResponse(
            sentiment_label=LABEL_MAPPING[predicted_idx],
            confidence=float(probs[predicted_idx]),
            probabilities=scores,
            aspects=aspects,
            emotions=emotions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))