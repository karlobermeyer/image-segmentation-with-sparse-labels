"""
Semantic segmentation ontology.
"""
from typing import Tuple

from core.identifiers import SemanticClassId


class SemanticClass:
    def __init__(
        self,
        id_: SemanticClassId,
        name: str,
        color_rgb: Tuple[int, int, int],
    ) -> None:
        self.id: SemanticClassId = id_
        self.name: str = name
        self.color_rgb: Tuple[int, int, int] = color_rgb
