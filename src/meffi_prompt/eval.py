from logging import getLogger
import pickle
import random
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.functional import softmax
from torch.optim import AdamW
from tqdm import tqdm


logger = getLogger(__name__)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    batch_processor: Callable,
    num_epochs: int = 5,
    lr: float = 3e-5,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    # set optimizer and device
    optimizer = AdamW(params=model.parameters(), lr=lr)
    model.to(device)

    with tqdm(total=num_epochs * len(train_loader)) as pbar:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                input_encodings, output_encodings = batch_processor(batch)
                loss = model(input_encodings, output_encodings).loss
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

            logger.info(
                "Epoch [{}/{}], Training Loss: {:.4f}.".format(
                    epoch + 1, num_epochs, epoch_loss
                )
            )
    return model


def train_and_eval(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    batch_processor: Callable,
    num_epochs: int = 5,
    lr: float = 3e-5,
    device: torch.device = torch.device("cpu"),
    label_column_name: str = "relation",
    tokenized_verbalizer: Dict[str, torch.Tensor] = None,
) -> Tuple[float]:
    # set optimizer and device
    optimizer = AdamW(params=model.parameters(), lr=lr)
    model.to(device)

    with tqdm(total=num_epochs * len(train_loader)) as pbar:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                input_encodings, output_encodings = batch_processor(batch)
                loss = model(input_encodings, output_encodings).loss
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

            logger.info(
                "Epoch [{}/{}], Training Loss: {:.4f}.".format(
                    epoch + 1, num_epochs, epoch_loss
                )
            )

            with torch.no_grad():
                model.eval()
                preds, labels = [], []
                for batch in tqdm(eval_loader):
                    batch.pop("target", None)
                    input_encodings = batch_processor(batch)
                    outputs = model(input_encodings)

                    batch_logits = torch.stack(outputs.scores, dim=-2).detach().cpu()
                    for logits in batch_logits:
                        preds.append(
                            predict_relation_from_logits(
                                logits=logits,
                                tokenized_verbalizer=tokenized_verbalizer,
                            )
                        )
                    labels += batch[label_column_name]

            positive_labels = set(tokenized_verbalizer.keys())
            positive_labels.discard("no_relation")
            positive_labels = list(positive_labels)

            micro_f1 = f1_score(
                labels,
                preds,
                labels=positive_labels,
                average="micro",
            )
            macro_f1 = f1_score(
                labels,
                preds,
                labels=positive_labels,
                average="macro",
            )
            logger.info(
                "Epoch [{}/{}], Validation micro-F1: {:.4f}, macro-F1: {:.4f}.".format(
                    epoch + 1, num_epochs, micro_f1, macro_f1
                )
            )

            cls_report = classification_report(labels, preds, digits=4, zero_division=0)
            with open("classification_report.txt", "a") as f:
                f.write(cls_report)

        # save labels and preds
        with open("labels.pkl", "wb") as fp:
            pickle.dump(labels, fp)
        with open("preds.pkl", "wb") as fp:
            pickle.dump(preds, fp)
    return micro_f1, macro_f1


def eval(
    model: nn.Module,
    eval_loader: DataLoader,
    batch_processor: Callable,
    device: torch.device = torch.device("cpu"),
    label_column_name: str = "relation",
    tokenized_verbalizer: Dict[str, torch.Tensor] = None,
) -> Tuple[float]:
    model.to(device)

    with torch.no_grad():
        model.eval()
        preds, labels = [], []
        for batch in tqdm(eval_loader):
            batch.pop("target", None)
            input_encodings = batch_processor(batch)
            outputs = model(input_encodings)

            batch_logits = torch.stack(outputs.scores, dim=-2).detach().cpu()
            for logits in batch_logits:
                preds.append(
                    predict_relation_from_logits(
                        logits=logits,
                        tokenized_verbalizer=tokenized_verbalizer,
                    )
                )
            labels += batch[label_column_name]

    positive_labels = set(tokenized_verbalizer.keys())
    positive_labels.discard("no_relation")
    positive_labels = list(positive_labels)

    micro_f1 = f1_score(
        labels,
        preds,
        labels=positive_labels,
        average="micro",
    )
    macro_f1 = f1_score(
        labels,
        preds,
        labels=positive_labels,
        average="macro",
    )
    logger.info(
        "Validation micro-F1: {:.4f}, macro-F1: {:.4f}.".format(micro_f1, macro_f1)
    )

    cls_report = classification_report(labels, preds, digits=4, zero_division=0)
    with open("classification_report.txt", "a") as f:
        f.write(cls_report)

    # save labels and preds
    with open("labels.pkl", "wb") as fp:
        pickle.dump(labels, fp)
    with open("preds.pkl", "wb") as fp:
        pickle.dump(preds, fp)
    return micro_f1, macro_f1


def predict_relation_from_logits(
    logits: torch.Tensor,
    tokenized_verbalizer: Dict[str, int],
) -> str:
    """Predict relation from output logits.

    Args:
        logits: A tensor of shape `[seq_len, vocab_size]` that corresponds to one decoded
        sequence.
        tokenized_verbalizer: A dictionary with keys being the relations and the values being
        a list of ids of relations.
        target_template: The target template for us to locate the decoded relation.
        min_id_of_sentinel: The minimum id of extra ids for T5 models.

    Returns:
        A relation name given by `str`.
    """
    relation_logits = logits[1:]
    relation_logits = softmax(relation_logits, dim=-1)
    score_per_relation: Dict[str, float] = {}
    for relation, tokenized in tokenized_verbalizer.items():
        valid_len = min(len(tokenized), len(relation_logits))
        probs = [relation_logits[i][tokenized[i]] for i in range(valid_len)]
        score_per_relation[relation] = sum(probs) / len(probs) if len(probs) > 0 else 0

    # take the ralation name with the highest likelihood
    # if more than one relations correspond to it, randomly select one as result
    max_score = max(score_per_relation.values())
    return random.choice([k for k, v in score_per_relation.items() if v == max_score])
