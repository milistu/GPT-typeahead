from pathlib import Path
from typing import Dict, List

from loguru import logger
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, pipeline

from .base import TellerBase
from .util import get_device, format_completion


class GPTeller(TellerBase):
    def __init__(
        self, device: str = None, cache_dir: Path = Path("./cache_dir")
    ) -> None:
        self.model_id = "openai-gpt"
        self.device = get_device if device is None else device
        self.cache_dir = cache_dir
        logger.info(f"Downloading weights to {cache_dir} directory.")

        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_id, cache_dir=self.cache_dir
        )
        self.model = OpenAIGPTLMHeadModel.from_pretrained(
            pretrained_model_name_or_path=self.model_id, cache_dir=self.cache_dir
        )
        logger.info(
            f"Using {self.model_id} for Query Completion with {self.model.num_parameters():_} parameters."
        )

        self.pipeline = pipeline(
            task="text-generation",
            tokenizer=self.tokenizer,
            model=self.model,
        )

    def complete(
        self,
        query: str,
        context: str = None,
        max_length: int = 10,
        num_return_sequences: int = 5,
        truncation: bool = True,
        num_beams: int = 5,
    ) -> List[Dict]:
        """
        num_beams - shoud be the same value as num_return_sequences to get most probable predictions always.
        fix diversity in compleations.
        """
        context = "" if context is None else f"[{context}] "
        completions = self.pipeline(
            context + query,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            truncation=truncation,
            num_beams=num_beams,
            do_sample=False,
        )

        return format_completion(compleations=completions, context=context)
