from typing import List, Dict, Any, Optional

from ..data import RCDataset


TEMPLATE = {
    "input": ["x", "[vN]", "eh", "[vN]", "<extra_id_0>", "[vN]", "et"],
    "target": ["<extra_id_0>", "r", "<extra_id_1>"],
}


class Prompt:
    def __init__(
        self,
        template: List[str] = TEMPLATE,
        model_name: str = "t5-large",
        soft_token_length: int = 0,
    ):
        """Given a prompt template, Initialize a prompting function that can be applied
        to dataset.
        """
        self.model_name = model_name
        self.template = template
        self.soft_token_length = soft_token_length

    @staticmethod
    def get_verbalized_relations(
        relations: List[str], language: str = "en"
    ) -> Dict[str, str]:
        """Get a mapping from original relation names to natural language relations, which
        usually involves replaceing "_" or "-" with space and extending "loc" to "location".
        Then translate those relation names to the target language if specified.
        """
        pass

    def apply_prompt(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply prompt on example-level, and we should use `map` to extend this function
        to dataset-level.

        After this operation, a new column "input" is created as the input tokens to the PLM,
        and a new column "target" is created as the target of the auto-regressive decoder.

        Example: given the first dev example with
        "token":
            ['At', 'the', 'same', 'time', ',', 'Chief', 'Financial', 'Officer', 'Douglas',
            'Flint', 'will', 'become', 'chairman', ',', 'succeeding', 'Stephen', 'Green',
            'who', 'is', 'leaving', 'to', 'take', 'a', 'government', 'job', '.'],
        we get "input":
            ['At', 'the', 'same', 'time', ',', 'Chief', 'Financial', 'Officer', 'Douglas',
            'Flint', 'will', 'become', 'chairman', ',', 'succeeding', 'Stephen', 'Green',
            'who', 'is', 'leaving', 'to', 'take', 'a', 'government', 'job', '.',
            'Douglas', 'Flint', '<v0>', '<v1>', '<v2>', '<extra_id_0>', 'chairman']
        and "target":
            ['<extra_id_0>', 'title', '<extra_id_1>'].
        """
        prompted_input, prompted_target = [], []

        # count of soft tokens <v0>, <v1>, <v2>, ...
        count_soft_token = 0

        # construct input sequence
        for token in self.template["input"]:
            if token == "x":
                prompted_input += example["token"]

            elif "<extra_id_>" in token:
                prompted_input.append(token)

            elif token == "[vN]":
                for _ in range(self.soft_token_length):
                    prompted_input.append("<v{}>".format(count_soft_token))
                    count_soft_token += 1

            elif token == "eh":
                start, end = example["subj_start"], example["subj_end"]
                entity = example["token"][start : (end + 1)]
                prompted_input += entity
            elif token == "et":
                start, end = example["obj_start"], example["obj_end"]
                entity = example["token"][start : (end + 1)]
                prompted_input += entity
            else:
                # additional real tokens
                prompted_input.append(token)

        # construct target sequence
        for token in self.template["target"]:
            if token == "r":
                prompted_target += example["verbalized_relation"].split(" ")
            else:
                prompted_target.append(token)

        example["input"], example["target"] = prompted_input, prompted_target
        return example

    def __call__(
        self,
        dataset: RCDataset,
        translate: bool = True,
        return_verbalizer: bool = False,
    ) -> RCDataset:
        """Transform original dataset to prompted dataset.
        First, we lemmatize (and translate if necessary) entity types (if provided) and
        relation names and get dictionary such as `verbalized_relations` and
        `verbalized_entity_types`;
        Second, we create a new column `verbalized_relation`
        for the dataset to store e.g. `"city of birth"` in case of `"'per:city_of_birth"`;
        New columns `verbalized_subj_type` and `verbalized_obj_type` are added similarly.
        Last, we construct input and target sequences using (1) the given template; (2) the
        verbalized entity type and relation name.
        """
        relations = list(dataset.label_to_id)
        verbalizer_lang = dataset.lang if translate else "en"
        verbalized_relations = self.get_verbalized_relations(relations, verbalizer_lang)
        dataset = dataset.add_column(
            "verbalized_relation",
            [verbalized_relations[relation] for relation in dataset["relation"]],
        )

        # apply prompt to get input and target sequences
        dataset = dataset.add_column("input", [""] * len(dataset))
        dataset = dataset.add_column("target", [""] * len(dataset))
        dataset = dataset.map(self.apply_prompt)

        if return_verbalizer:
            return dataset, verbalized_relations
        return dataset


def get_num_soft_tokens(
    template: Dict[str, List[str]], soft_token_length: int = 0
) -> int:
    """Given a template, the number of soft tokens equals the number of `[vN]`s times
    `soft_token_length`.
    """
    return template["input"].count("[vN]") * soft_token_length


def get_max_decode_length(
    template: Dict[str, List[str]], max_relation_length: int = 5
) -> int:
    template = template["target"]
    # start from 1 since "</s>" should be considered as EOS token
    max_decode_length = 1
    for token in template:
        if token == "r":
            max_decode_length += max_relation_length
        else:
            max_decode_length += 1
    return max_decode_length
