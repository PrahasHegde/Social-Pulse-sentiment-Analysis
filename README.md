# ⚡ Social Pulse — Next-Gen AI Sentiment & Intelligence Engine

A production-grade, microservice-based Sentiment Analysis and Social Intelligence system built with **FastAPI**, **RoBERTa Transformers**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![Transformers](https://img.shields.io/badge/HuggingFace-RoBERTa-yellow)

---

## 🌟 Features

* **🎭 Real-Time Sentiment Classification:** Powered by a domain-optimized RoBERTa model (`twitter-roberta-base-sentiment-latest`) for state-of-the-art accuracy on social text.
* **🏷️ Aspect-Based Extraction (ABSA):** Detects domain topics (*UI/Product*, *Performance*, *Pricing*, *Customer Service*) within posts.
* **📊 Multi-Dimensional Emotion Profiling:** Radar visualization across 6 core emotional dimensions (*Joy, Anger, Sadness, Surprise, Disgust, Trust*).
* **⚡ High-Throughput REST API:** Built using **FastAPI** with structured Pydantic data validation and fast inference execution.
* **📈 Interactive Streamlit Dashboard:** Features single-post deep inspection, batch CSV processing, and hierarchical Plotly visual analytics.

---

## 🏗️ Architecture

```
┌──────────────────────────────┐
│  Streamlit Frontend Dashboard│  (Visual UI, Plotly Charts)
└──────────────┬───────────────┘
               │ HTTP / JSON
               ▼
┌──────────────────────────────┐
│    FastAPI Microservice      │  (Data Validation & API Routes)
└──────────────┬───────────────┘
               │ Cleaned Text
               ▼
┌──────────────────────────────┐
│ Preprocessing & Transformer  │  (RoBERTa Sentiment + Emotion Engine)
└──────────────────────────────┘
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment

```bash
# Clone repository
git clone https://github.com/your-username/social-sentiment-ai.git
cd social-sentiment-ai

# Create & activate virtual environment
python -m venv venv

# On Mac/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Launch the System

Run both services in separate terminal windows:

#### Terminal 1 — Start FastAPI Engine:
```bash
uvicorn app:app --reload
```
> The API will start at `[http://127.0.0.1:8000](http://127.0.0.1:8000)`. Access `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)` for interactive API documentation.

#### Terminal 2 — Start Streamlit Frontend:
```bash
streamlit run dashboard.py
```
> The Dashboard will open automatically at `http://localhost:8501`.

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check endpoint for system monitoring. |
| `POST` | `/predict` | Main inference endpoint. Receives text, returns sentiment probabilities, aspects, and emotion scores. |

### Sample Request:
```json
POST /predict
{
  "text": "I absolutely love the new UI, but the battery drain issue is unacceptable! @support"
}
```

### Sample Response:
```json
{
  "sentiment_label": "negative",
  "confidence": 0.884,
  "probabilities": {
    "negative": 0.884,
    "neutral": 0.082,
    "positive": 0.034
  },
  "aspects": ["Product/UI", "Performance"],
  "emotions": {
    "Joy": 0.03,
    "Anger": 0.62,
    "Sadness": 0.44,
    "Surprise": 0.10,
    "Disgust": 0.53,
    "Trust": 0.24
  }
}
```

---

## 📄 License
This project is open-source under the MIT License.