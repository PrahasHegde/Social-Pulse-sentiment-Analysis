import re

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