import re
from typing import Dict, List

import torch


def format_completion(compleations: List[Dict]) -> List[str]:
    return [clean_spaces(compleation) for compleation in compleations]


def clean_spaces(text):
    # Replace one or more spaces with a single space
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    return cleaned_text


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
