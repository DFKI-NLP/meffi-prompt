from typing import Dict

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput


class Model(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        max_decode_length: int = 10,
        tokenizer: AutoTokenizer = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_decode_length = max_decode_length

        self.vocab_size = tokenizer.vocab_size
        self.tokenizer_size = len(tokenizer)
        self.num_soft_tokens = self.tokenizer_size - self.vocab_size

    def forward(
        self,
        input_encodings: Dict[str, torch.Tensor],
        target_encodings: Dict[str, torch.Tensor],
    ) -> Seq2SeqLMOutput:
        pass

    def token_id_to_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids (both hard and soft) to same-length embeddings as model."""
        if self.num_soft_tokens == 0:
            return self.model.shared(input_ids.to(self.device))

        # mask hard token (with id < 32100) as True and soft as False
        hard_token_mask = (
            (input_ids < self.vocab_size)
            .unsqueeze(-1)
            .expand(
                (input_ids.size(0), input_ids.size(1), self.model.config.hidden_size)
            )
        ).to(self.device)

        # get old embedding, treating soft labels as <unk> to avoid out-of-index error
        _input_ids = input_ids.clone().apply_(lambda x: x if x < self.vocab_size else 2)
        raw_embedding = self.model.shared(_input_ids.to(self.device))

        # get new embedding for soft tokens
        _input_ids = input_ids.clone().apply_(
            lambda x: 0 if x < self.vocab_size else x - self.vocab_size
        )
        new_embedding = self.soft_embedding(_input_ids.to(self.device))

        return torch.where(hard_token_mask, raw_embedding, new_embedding)
