import re
import emoji

def normalize_amharic(text):
    # Example: remove extra spaces, normalize punctuation, etc.
    text = re.sub(r'[፡።፣፤፥፦፧፨]', ' ', text)  # Amharic punctuation
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def tokenize_amharic(text):
    # Simple whitespace tokenizer (replace with better if available)
    return text.split()

def preprocess_amharic_text(text):
    text = normalize_amharic(text)
    text = remove_emojis(text)
    tokens = tokenize_amharic(text)
    return ' '.join(tokens)
