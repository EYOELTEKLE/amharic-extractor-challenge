import os
from typing import List, Tuple

def parse_conll(filepath: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Parse a CoNLL-format file into lists of tokens and labels.
    Returns: sentences, ner_tags
    """
    sentences = []
    labels = []
    with open(filepath, encoding='utf-8') as f:
        tokens = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[-1])
        if tokens:
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels

def build_label_maps(ner_tags: List[List[str]]):
    unique_labels = sorted(set(tag for tags in ner_tags for tag in tags))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label
