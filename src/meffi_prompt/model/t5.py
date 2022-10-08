from typing import Dict

import torch
from torch import nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from .base import Model


class T5Model(Model):
    def __init__(
        self,
        model_name_or_path: str = "t5-large",
        max_decode_length: int = 10,
        tokenizer: T5Tokenizer = None,
    ):
        super().__init__(model_name_or_path, max_decode_length, tokenizer)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        # fix the size of embedding for hard tokens
        self.model.resize_token_embeddings(self.vocab_size)

        # build another embedding for soft tokens
        if self.num_soft_tokens > 0:
            self.soft_embedding = nn.Embedding(
                self.num_soft_tokens, self.model.config.hidden_size
            )
            self.soft_embedding.weight.data.normal_(0, 1)

    def forward(
        self,
        input_encodings: Dict[str, torch.Tensor],
        target_encodings: Dict[str, torch.Tensor] = None,
    ) -> Seq2SeqLMOutput:
        self.device = self.model.device

        # training scenario
        if target_encodings is not None:
            inputs_embeds = self.token_id_to_embedding(input_encodings["input_ids"])
            attention_mask = input_encodings["attention_mask"].to(self.device)
            labels = target_encodings["input_ids"].to(self.device)

            return self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        # inference scenario
        else:
            inputs_embeds = self.token_id_to_embedding(input_encodings["input_ids"])
            attention_mask = input_encodings["attention_mask"].to(self.device)

            # .generate() method does not support inputs_embeds directly as params
            encoder_outputs = self.model.base_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )

            return self.model.generate(
                encoder_outputs=encoder_outputs,
                max_length=self.max_decode_length,
                return_dict_in_generate=True,
                output_scores=True,
            )
