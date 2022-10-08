import logging
from typing import Union, List, Dict, Any, Callable, Optional

import datasets
import random
import torch


logger = logging.getLogger(__name__)

_TEXT_COLUMN_NAME = "token"
_LABEL_COLUMN_NAME = "relation"


class RCDataset(torch.utils.data.Dataset):
    """A general relation classification (RC) dataset from the raw dataset."""

    def __init__(
        self,
        data_file: str,
        text_column_name: str = _TEXT_COLUMN_NAME,
        label_column_name: str = _LABEL_COLUMN_NAME,
        negative_label: Optional[str] = None,
    ):
        """
        Args:
            data_file: Path to the .json file for the split of data.
            text_colomn_name: The name of the column for the text, e.g. "token".
            label_column_name: The name of the column for the label, e.g. "relation".
            negative_label: The negative relation name, e.g. "no_relation".
        """
        super().__init__()
        self.dataset = datasets.load_dataset(
            "json", data_files=data_file, split="train"
        )
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.negative_label = negative_label

        self.label_to_id = self._get_label_to_id()
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.lang, self.orig_lang = "en", "en"

    def __getitem__(self, index: Union[int, List[int], torch.Tensor]):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def add_column(self, new_column_name: str, new_column: List[Any]):
        """Reload the `add_column` function from torch.utils.data.Dataset, so that this
        wrapper class can simply call `object = object.add_column(...)` which makes it
        look like performing the add-column operation directly on a torch Dataset, instead
        of `object.dataset = object.dataset.add_column(...)`.
        """
        self.dataset = self.dataset.add_column(new_column_name, new_column)
        return self

    def map(self, function: Callable):
        """Reload the `map` function from torch.utils.data.Dataset, so that this wrapper
        class can simply call `object = object.map(...)` instead of the clumsy way
        `object.dataset = object.dataset.map(...)`.
        """
        self.dataset = self.dataset.map(function)
        return self

    def _get_label_to_id(self) -> Dict[str, int]:
        """Get a dict of the class-id mapping."""
        label_list = list(set(self.dataset[self.label_column_name]))
        label_list.sort()
        return {label: i for i, label in enumerate(label_list)}

    def add_column_for_label_id(self, new_column_name: str = "relation_id") -> None:
        """Add a new column to store the (relation) class ids mapped from the relation names."""
        new_column = [
            self.label_to_id[label] for label in self.dataset[self.label_column_name]
        ]
        self.dataset = self.dataset.add_column(new_column_name, new_column)

    @staticmethod
    def convert_special_tokens(
        example: Dict[str, Any],
        text_column_name: str = "token",
        special_tokens_dict: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """An example-level processor to convert special tokens to natural language tokens, e.g.
        "-lsb-" to "[".
        """
        converted_tokens = []
        for token in example[text_column_name]:
            if token.lower() in special_tokens_dict:
                converted_tokens.append(special_tokens_dict[token.lower()])
            else:
                converted_tokens.append(token)
        example[text_column_name] = converted_tokens
        return example


class RCFewShotDataset(RCDataset):
    """Few-shot version of the general RC dataset."""

    def __init__(
        self,
        data_file: str,
        text_column_name: str = _TEXT_COLUMN_NAME,
        label_column_name: str = _LABEL_COLUMN_NAME,
        negative_label: Optional[str] = None,
        kshot: int = 5,
        include_negative: bool = True,
    ):
        super().__init__(data_file, text_column_name, label_column_name, negative_label)
        self.kshot = kshot
        self.include_negative = include_negative

        self.class_indices = self._get_indices_per_class()
        self.num_examples = {k: len(v) for k, v in self.class_indices.items()}
        # print the relation names in descending order of number of examples
        logger.info("Number of examples per class:", self.num_examples)

        self.sampled_indices = self._sample_indices()
        self.dataset = self.dataset.select(self.sampled_indices).flatten_indices()
        logger.info("{} Examples in the few-shot dataset.".format(len(self.dataset)))

    def _get_indices_per_class(self) -> Dict[Any, List[int]]:
        """For each class (in this case "relation"), maintain a list of indices where the examples
        are from this class, so that we can easily sample from given classes later."""
        class_indices: Dict[str, List[int]] = {}
        for idx, example in enumerate(self.dataset):
            label = example[self.label_column_name]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def _sample_indices(self) -> List[int]:
        """Sample the indices in the dataset: For each of the `N` classes, sample `K` indices."""
        # get a list of valid (i.e. with sufficient size and positive if specified) classes
        if not self.include_negative:
            self.class_indices.pop(self.negative_label, None)

        # self.sampled_classes = random.sample(list(self.label_to_id.keys()), k=self.nway)
        # sample K-shots for each sampled class
        sampled_indices: Dict[int] = []
        for sampled_class in list(self.class_indices.keys()):
            # if this class has sufficient examples, choose k of them; otherwise take all of them.
            sampled_indices.extend(
                random.sample(
                    self.class_indices[sampled_class],
                    k=min(len(self.class_indices[sampled_class]), self.kshot),
                )
            )
        return sampled_indices

    def __len__(self):
        return len(self.dataset)
