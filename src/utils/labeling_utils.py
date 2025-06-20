import pandas as pd
from src.utils.preprocessing import tokenize_amharic
from typing import List, Tuple

def load_telegram_messages_xlsx(path: str, n: int = 40) -> pd.DataFrame:
    """Load n messages from the specified Excel file."""
    df = pd.read_excel(path)
    sampled = df[['Message']].dropna().sample(n=n, random_state=42).reset_index(drop=True)
    return sampled

def tokenize_message(message: str) -> List[str]:
    """Tokenize an Amharic message."""
    return tokenize_amharic(str(message))

def label_tokens(tokens: List[str], labels: List[str]) -> List[Tuple[str, str]]:
    """Pair tokens with their labels."""
    assert len(tokens) == len(labels), "Tokens and labels must be the same length."
    return list(zip(tokens, labels))

def save_conll_format(labeled_messages: List[List[Tuple[str, str]]], output_path: str):
    """Save labeled tokens in CoNLL format to a text file."""
    lines = []
    for message in labeled_messages:
        for token, label in message:
            lines.append(f"{token} {label}")
        lines.append("")  # Blank line between messages
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
