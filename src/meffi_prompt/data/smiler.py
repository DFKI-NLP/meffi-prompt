from logging import getLogger
from typing import Optional

from .base import RCDataset, RCFewShotDataset


logger = getLogger(__name__)

_TEXT_COLUMN_NAME = "token"
_LABEL_COLUMN_NAME = "relation"
_NEGATIVE_LABEL = "no_relation"


class SmilerDataset(RCDataset):
    """The Smiler dataset from the raw data file(s)."""

    def __init__(
        self,
        data_file: str,
        text_column_name: str = _TEXT_COLUMN_NAME,
        label_column_name: str = _LABEL_COLUMN_NAME,
        negative_label: Optional[str] = _NEGATIVE_LABEL,
    ):
        """
        Args:
            data_file: Path to the .json file for the split of data.
            text_colomn_name: The name of the column for the text.
            label_column_name: The name of the column for the label.
        """
        super().__init__(data_file, text_column_name, label_column_name, negative_label)
        self.add_column_for_label_id(new_column_name="relation_id")
        self.lang = self.source_lang = self.dataset["lang"][0]


class SmilerFewShotDataset(RCFewShotDataset, SmilerDataset):
    """Few-shot version of the SMiLER dataset."""

    def __init__(
        self,
        data_file: str,
        kshot: int = 5,
        include_negative: bool = True,
    ):
        super().__init__(
            data_file=data_file,
            text_column_name=_TEXT_COLUMN_NAME,
            label_column_name=_LABEL_COLUMN_NAME,
            negative_label=_NEGATIVE_LABEL,
            kshot=kshot,
            include_negative=include_negative,
        )
