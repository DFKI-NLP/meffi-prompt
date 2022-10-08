import json
import re
from typing import List, Dict, Any

import pandas as pd


_LANGUAGES = [
    "ar",
    "de",
    "en",
    "es",
    "fa",
    "fr",
    "it",
    "ko",
    "nl",
    "pl",
    "pt",
    "ru",
    "sv",
    "uk",
]


def preprocess_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Tokenize the "text" column and compute the entity start and end positions."""
    sentence = example["text"]
    if "</e2>" not in example["text"]:
        sentence = sentence.split("<e2>")[0] + "<e2>" + example["entity_2"] + "</e2>."
    if "</e1>" not in example["text"]:
        sentence = sentence.split("<e1>")[0] + "<e1>" + example["entity_1"] + "</e1>."

    sentence = (
        sentence.replace("<e1>", " subjstart ")
        .replace("</e1>", " subjend ")
        .replace("<e2>", " objstart ")
        .replace("</e2>", " objend ")
    )
    token = re.findall(r"[\w]+|[^\s\w]", sentence)
    subj_start, subj_end, obj_start, obj_end = (
        token.index("subjstart"),
        token.index("subjend"),
        token.index("objstart"),
        token.index("objend"),
    )
    for marker in ["subjstart", "subjend", "objstart", "objend"]:
        if marker in token:
            token.remove(marker)

    # calibrate the entity spans after removal of markers
    if subj_start < obj_start:
        subj_end -= 2
        obj_start -= 2
        obj_end -= 4
    else:
        obj_end -= 2
        subj_start -= 2
        subj_end -= 4

    return {
        "lang": example["lang"],
        "id": example["id"],
        "subject": str(example["entity_1"]),
        "object": str(example["entity_2"]),
        "relation": str(example["label"]),
        "token": token,
        "subj_start": subj_start,
        "subj_end": subj_end,
        "obj_start": obj_start,
        "obj_end": obj_end,
    }


def preprocess_data_file(data_file: str) -> None:
    """Read the original .tsv files and write the processed dataset into json."""
    df = pd.read_csv(data_file, sep="\t")
    dataset: List[Dict] = df.to_dict(orient="records")
    dataset = [preprocess_example(example) for example in dataset]

    with open(data_file.replace(".tsv", ".json"), "w") as fw:
        for example in dataset:
            fw.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    for lang in _LANGUAGES:
        for split in ["train", "test"]:
            data_file = "{}_corpora_{}.tsv".format(lang, split)
            preprocess_data_file(data_file)
