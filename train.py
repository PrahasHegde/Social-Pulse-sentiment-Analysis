from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def save_model():
    print(f"Downloading and saving model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    tokenizer.save_pretrained("./model_assets")
    model.save_pretrained("./model_assets")
    print("Successfully saved model assets to ./model_assets")

if __name__ == "__main__":
    save_model()