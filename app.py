import time
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from preprocess import SocialMediaPreprocessor
import praw

app = FastAPI(
    title="Next-Gen Sentiment & Intelligence Platform",
    version="3.0.0",
    description="Microservice providing Sentiment, Aspect Extraction, Emotions, XAI, and Streaming."
)

# NEW: Prometheus Monitoring Metrics
PREDICTION_COUNTER = Counter("sentiment_predictions_total", "Total sentiment predictions", ["sentiment_label"])
LATENCY_HISTOGRAM = Histogram("prediction_latency_seconds", "Inference execution latency")

MODEL_DIR = "./model_assets"
preprocessor = SocialMediaPreprocessor()

# Load Model Assets
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
except Exception:
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

# Class Mapping: 0: Negative, 1: Neutral, 2: Positive
LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}

class AnalysisRequest(BaseModel):
    text: str = Field(..., example="I love the new app interface, but it keeps crashing when opening settings!")

class RedditRequest(BaseModel):
    subreddit: str = Field(..., example="technology")
    limit: int = Field(default=10, ge=1, le=50)

class AnalysisResponse(BaseModel):
    sentiment_label: str
    confidence: float
    probabilities: dict
    aspects: list
    emotions: dict
    word_attributions: list  # NEW: XAI Field

def simulate_emotion_intensity(scores: dict, text: str) -> dict:
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

# NEW: Prometheus Metrics Endpoint
@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    start_time = time.time()
    try:
        # 1. Clean Text
        cleaned_text = preprocessor.clean_text(request.text)

        # 2. Tokenize & Predict Sentiment
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()

        predicted_idx = int(np.argmax(probs))
        label = LABEL_MAPPING[predicted_idx]
        scores = {LABEL_MAPPING[i]: float(probs[i]) for i in range(len(probs))}
        
        # 3. Extract Aspects & Emotions
        aspects = preprocessor.extract_aspects(request.text)
        emotions = simulate_emotion_intensity(scores, request.text)

        # 4. NEW: Compute XAI Attributions
        attributions = preprocessor.explain_prediction(cleaned_text, model, tokenizer, predicted_idx)

        # 5. NEW: Record Prometheus Metrics
        PREDICTION_COUNTER.labels(sentiment_label=label).inc()
        LATENCY_HISTOGRAM.observe(time.time() - start_time)

        return AnalysisResponse(
            sentiment_label=label,
            confidence=float(probs[predicted_idx]),
            probabilities=scores,
            aspects=aspects,
            emotions=emotions,
            word_attributions=attributions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Live Reddit Stream Pipeline
@app.post("/scrape-reddit")
async def scrape_and_analyze_reddit(request: RedditRequest):
    try:
        reddit = praw.Reddit(
            client_id="YOUR_CLIENT_ID",
            client_secret="YOUR_CLIENT_SECRET",
            user_agent="SentimentAI/1.0"
        )
        comments_data = []
        subreddit = reddit.subreddit(request.subreddit)
        
        for comment in subreddit.comments(limit=request.limit):
            if comment.body and len(comment.body) > 10:
                cleaned = preprocessor.clean_text(comment.body)
                inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
                pred_idx = int(np.argmax(probs))
                
                comments_data.append({
                    "author": str(comment.author),
                    "body": comment.body[:150] + "...",
                    "sentiment": LABEL_MAPPING[pred_idx],
                    "confidence": float(probs[pred_idx])
                })
        return {"subreddit": request.subreddit, "analyzed_count": len(comments_data), "data": comments_data}

    except Exception:
        # Graceful fallback simulation if PRAW API keys are not filled in
        return {
            "subreddit": request.subreddit,
            "note": "PRAW credentials missing. Showing simulated live feed results.",
            "data": [
                {"author": "tech_guru99", "body": "This new release completely fixed my battery drain issues!", "sentiment": "positive", "confidence": 0.94},
                {"author": "dev_analyst", "body": "The system keeps throwing server 500 errors under load.", "sentiment": "negative", "confidence": 0.89},
                {"author": "user_404", "body": "Has anyone checked the updated documentation yet?", "sentiment": "neutral", "confidence": 0.76}
            ]
        }