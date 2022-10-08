from typing import List, Dict, Any

from .base import Prompt, TEMPLATE
from .verbalizers import SMILER_VERBALIZERS


class SmilerPrompt(Prompt):
    """Prompt class for SMiLER dataset."""

    def __init__(
        self,
        template: Dict[str, Dict[str, List[str]]] = TEMPLATE,
        model_name: str = "t5-large",
        soft_token_length: int = 0,
    ):
        super().__init__(template, model_name, soft_token_length)

    @staticmethod
    def get_verbalized_relations(
        relations: List[str], language: str = "en"
    ) -> Dict[str, str]:
        """Get a mapping from original relation names to natural language relations, which
        usually involves replaceing "-" with space and extending "loc" to "location". Then
        translate those relation names to the target language.
        """
        # verbalized_relations = {
        #     x: x.replace("_", " ")
        #     .replace("-", " ")
        #     .replace("org", "organization")
        #     .replace("loc", "location")
        #     .replace("edu", "education")
        #     .replace("is where", "located in")
        #     for x in relations
        # }
        verbalized_relations = {x: SMILER_VERBALIZERS[language][x] for x in relations}
        return verbalized_relations
