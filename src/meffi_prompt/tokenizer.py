from typing import Tuple, List, Dict, Any, Union

import torch
from transformers import AutoTokenizer


class Tokenizer:
    def __init__(
        self,
        tokenizer_name_or_path: str = "t5-large",
        max_length: int = 256,
        num_soft_tokens: int = 0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length
        self.num_soft_tokens = num_soft_tokens

        # add soft tokens to vocab for T5
        if "t5" in tokenizer_name_or_path:
            soft_tokens = ["<v{}>".format(i) for i in range(self.num_soft_tokens)]
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": soft_tokens}
            )

    def __call__(self, tokens: Union[str, List[str]]) -> torch.Tensor:
        """Call the tokenizer to tokenize the text."""
        if isinstance(tokens, str) and " " in tokens:
            tokens = tokens.split(" ")

        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=False,
        )
        return tokenized


class BatchTokenizer(Tokenizer):
    def __init__(
        self,
        tokenizer_name_or_path: str = "t5-large",
        max_length: int = 256,
        num_soft_tokens: int = 0,
    ):
        super().__init__(tokenizer_name_or_path, max_length, num_soft_tokens)

    def __call__(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Call the tokenizer to tokenize the batch."""
        input_encodings = self.tokenizer(
            batch["input"],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=False,
        )

        if "target" in batch:
            target_encodings = self.tokenizer(
                batch["target"],
                is_split_into_words=True,
                padding=True,
                return_tensors="pt",
                return_offsets_mapping=False,
            )
            return input_encodings, target_encodings
        return input_encodings
