# --- Preprocessing Logic (from data_pipeline.ipynb) ---
import emoji
import re

def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    
    emoji_pattern = emoji.get_emoji_regexp().pattern
    GARBAGE_PATTERN = r"array\(|dtype=|\{'text':|\[|\]"
    
    # Remove emojis
    text = re.sub(emoji_pattern, "", text)
    # Remove garbage patterns
    text = re.sub(GARBAGE_PATTERN, "", text)
    # Remove Tashkeel/Tatweel
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)
    # Normalize Alef
    text = re.sub(r"[إأآا]", "ا", text)
    # Normalize Yeh
    text = re.sub(r"ى", "ي", text)
    # Normalize Waw
    text = re.sub(r"ؤ", "و", text)
    # Normalize Yeh Hamza
    text = re.sub(r"ئ", "ي", text)
    # Normalize Teh Marbuta
    text = re.sub(r"ة", "ه", text)
    # Keep only Arabic, numbers, punctuation (normalize non-arabic chars to space)
    text = re.sub(r"[^\u0600-\u06FF0-9\s.,؟!]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text
