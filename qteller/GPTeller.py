from pathlib import Path
from typing import List, Tuple

import torch
from loguru import logger
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

from .base import TellerBase
from .util import format_completion


class GPTeller(TellerBase):
    def __init__(
        self, device: str = None, cache_dir: Path = Path("./cache_dir")
    ) -> None:
        self.model_id = "openai-gpt"
        # self.device = get_device() if device is None else device
        self.cache_dir = cache_dir
        logger.info(f"Downloading weights to {cache_dir} directory.")

        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            cache_dir=self.cache_dir,
        )
        self.model = OpenAIGPTLMHeadModel.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            cache_dir=self.cache_dir,
        )
        logger.info(f"Running device: {self.model.device}.")
        logger.info(
            f"Using {self.model_id} for Query Completion with {self.model.num_parameters():_} parameters."
        )

    def encode(self, query: str, context: str) -> Tuple[torch.Tensor, int]:
        if context is None:
            tokenized_input = self.tokenizer.encode(query, return_tensors="pt")
            context_len = 0
        else:
            tokenized_query = self.tokenizer.encode(query, return_tensors="pt")
            tokenized_context = self.tokenizer.encode(context, return_tensors="pt")
            tokenized_input = torch.cat((tokenized_context, tokenized_query), dim=1)
            context_len = tokenized_context.shape[-1]

        return tokenized_input, context_len

    def decode(self, outputs: torch.Tensor, context_len: int) -> List[str]:
        return [
            self.tokenizer.decode(output[context_len:], skip_special_tokens=True)
            for output in outputs
        ]

    def complete(
        self,
        query: str,
        context: str = None,
        max_length: int = 3,
        num_return_sequences: int = 5,
        early_stopping: bool = True,
        num_beams: int = 5,
        num_beam_groups: int = 5,
        no_repeat_ngram_size: int = 2,
        diversity_penalty: float = 1.0,
    ) -> List[str]:
        """
        num_beams - shoud be the same value as num_return_sequences to get most probable predictions always.
        fix diversity in compleations.
        """
        tokenized_input, context_len = self.encode(query, context)

        outputs = self.model.generate(
            inputs=tokenized_input,
            max_length=len(tokenized_input[0]) + max_length,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            diversity_penalty=diversity_penalty,
        )
        completions = self.decode(outputs, context_len)
        return format_completion(compleations=completions)
