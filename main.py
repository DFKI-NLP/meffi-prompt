import logging
import json

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from meffi_prompt.utils import resolve_relative_path, seed_everything, aggregate_batch
from meffi_prompt.data import SmilerDataset
from meffi_prompt.prompt import SmilerPrompt, get_num_soft_tokens, get_max_decode_length
from meffi_prompt.model import T5Model
from meffi_prompt.tokenizer import BatchTokenizer
from meffi_prompt.eval import train_and_eval


logger = logging.getLogger(__name__)

# in-order + w/o reversed
template = {
    "input": ["x", "[vN]", "eh", "[vN]", "<extra_id_0>", "[vN]", "et"],
    "target": ["<extra_id_0>", "r", "<extra_id_1>"],
}

# # post-order
# template = {
#     "input": ["x", "[vN]", "eh", "[vN]", "et", "[vN]", "<extra_id_0>"],
#     "target": ["<extra_id_0>", "r", "<extra_id_1>"],
# }


@hydra.main(config_name="config", config_path="configs")
def main(cfg: DictConfig) -> None:
    """
    Conducts evaluation given the configuration.
    Args:
        cfg: Hydra-format configuration given in a dict.
    """
    resolve_relative_path(cfg)
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)
    device = (
        torch.device("cuda", cfg.cuda_device)
        if cfg.cuda_device > -1
        else torch.device("cpu")
    )

    # get raw dataset and do simple pre-processing such as convert special tokens
    train_dataset = SmilerDataset(cfg.train_file)
    eval_dataset = SmilerDataset(cfg.eval_file)

    # transform to prompted dataset, with appended inputs and verbalized labels
    prompt = SmilerPrompt(
        template=template,
        model_name=cfg.model,
        soft_token_length=cfg.soft_token_length,
    )
    train_dataset, verbalizer = prompt(
        train_dataset, translate=cfg.translate, return_verbalizer=True
    )
    eval_dataset = prompt(eval_dataset, translate=cfg.translate)

    # set dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=aggregate_batch,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=aggregate_batch,
    )

    # instantiate tokenizer and model
    num_soft_tokens = get_num_soft_tokens(template, cfg.soft_token_length)
    batch_processor = BatchTokenizer(
        tokenizer_name_or_path=cfg.model,
        max_length=cfg.max_length,
        num_soft_tokens=num_soft_tokens,
    )

    tokenized_verbalizer = {
        k: batch_processor.tokenizer(v, add_special_tokens=False)["input_ids"]
        for k, v in verbalizer.items()
    }
    max_relation_length = max([len(v) for v in tokenized_verbalizer.values()])
    max_decode_length = get_max_decode_length(template, max_relation_length)
    logger.info("Max decode length: {}.".format(max_decode_length))

    model = T5Model(
        cfg.model,
        max_decode_length=max_decode_length,
        tokenizer=batch_processor.tokenizer,
    )

    micro_f1, macro_f1 = train_and_eval(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        batch_processor=batch_processor,
        num_epochs=cfg.num_epochs,
        lr=cfg.lr,
        device=device,
        label_column_name=train_dataset.label_column_name,
        tokenized_verbalizer=tokenized_verbalizer,
    )
    logger.info(
        "Evaluation micro-F1: {:.4f}, macro_f1: {:.4f}.".format(micro_f1, macro_f1)
    )
    # save evaluation results to json
    with open("./results.json", "w") as f:
        json.dump({"micro_f1": micro_f1, "macro_f1": macro_f1}, f, indent=4)


if __name__ == "__main__":
    main()
