# visual_patterns.py

from dataclasses import dataclass
from typing import List, Set, Tuple
from enum import Enum, auto

class PatternType(Enum):
    ALIGNED_PAIR = auto()
    ALIGNED_TRIPLE = auto()
    BOX_PATTERN = auto()
    LINE_PATTERN = auto()
    CROSS_PATTERN = auto()
    CHAIN_PATTERN = auto()

@dataclass
class VisualGroup:
    pattern_type: PatternType
    cells: List[Tuple[int, int]]
    candidates: Set[int]
    strength: float
    
    def get_description(self) -> str:
        """Generate a basic description of the pattern."""
        cells_desc = ", ".join(f"({r+1},{c+1})" for r, c in self.cells)
        candidates_desc = ", ".join(map(str, sorted(self.candidates)))
        return f"{self.pattern_type.name} pattern in cells {cells_desc} with candidates {candidates_desc}"