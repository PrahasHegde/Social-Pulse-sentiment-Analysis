import re
import torch
import numpy as np

class SocialMediaPreprocessor:
    def __init__(self):
        self.user_regex = re.compile(r"@\w+")
        self.url_regex = re.compile(r"https?://\S+|www\.\S+")
        self.hashtag_regex = re.compile(r"#(\w+)")

    def clean_text(self, text: str) -> str:
        """Standardizes social media noise like URLs, handles, and hashtags."""
        text = self.url_regex.sub("http", text)
        text = self.user_regex.sub("@user", text)
        text = self.hashtag_regex.sub(r"\1", text)
        return " ".join(text.split())

    def extract_aspects(self, text: str) -> list:
        """Simple rule-based keyword extractor for Aspect-Based Analysis."""
        text_lower = text.lower()
        aspect_keywords = {
            "Product/UI": ["ui", "design", "app", "feature", "interface", "update", "screen", "look"],
            "Performance": ["fast", "slow", "lag", "bug", "crash", "speed", "performance", "smooth"],
            "Customer Service": ["support", "service", "help", "staff", "team", "agent", "response"],
            "Pricing/Value": ["price", "cost", "cheap", "expensive", "money", "value", "subscription"]
        }
        
        found_aspects = []
        for aspect, keywords in aspect_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_aspects.append(aspect)
                
        return found_aspects if found_aspects else ["General"]

    def explain_prediction(self, text: str, model, tokenizer, target_idx: int) -> list:
        """NEW: Calculates token-level feature importance attribution using gradient norms (XAI)."""
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            embeddings = model.get_input_embeddings()(inputs["input_ids"])
            embeddings.retain_grad()

            outputs = model(inputs_embeds=embeddings)
            logits = outputs.logits[0]
            
            # Compute gradient relative to predicted target class
            loss = logits[target_idx]
            loss.backward()
            
            grads = embeddings.grad[0]
            attribution_scores = grads.norm(dim=1).detach().numpy()
            
            if attribution_scores.max() > 0:
                attribution_scores = attribution_scores / attribution_scores.max()

            input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            word_attributions = []
            
            for tok, score in zip(input_tokens, attribution_scores):
                clean_tok = tok.replace("Ġ", "").replace(" ", "")
                if clean_tok not in ["[CLS]", "[SEP]", "<s>", "</s>", "<pad>", ""]:
                    word_attributions.append({"token": clean_tok, "score": float(score)})

            return word_attributions
        except Exception:
            # Safe fallback if gradient extraction encounters unmapped tensor shapes
            tokens = text.split()
            return [{"token": t, "score": 0.5} for t in tokens]