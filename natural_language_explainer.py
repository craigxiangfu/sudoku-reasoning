# natural_language_explainer.py

from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import random

from visual_patterns import VisualGroup, PatternType
from strategies import Strategy, Cell, SolveStep

class SpatialReference:
    """Helper class for generating natural spatial references."""
    
    @staticmethod
    def describe_location(row: int, col: int) -> str:
        """Generate a natural description of a cell's location."""
        box_row = row // 3
        box_col = col // 3
        
        # Box descriptions
        box_descriptions = {
            (0, 0): "top-left",
            (0, 1): "top-middle",
            (0, 2): "top-right",
            (1, 0): "middle-left",
            (1, 1): "center",
            (1, 2): "middle-right",
            (2, 0): "bottom-left",
            (2, 1): "bottom-middle",
            (2, 2): "bottom-right"
        }
        
        # Position within box
        within_box_row = row % 3
        within_box_col = col % 3
        position_desc = ""
        if within_box_row == 0:
            position_desc = "top"
        elif within_box_row == 2:
            position_desc = "bottom"
        if within_box_col == 0:
            position_desc += " left"
        elif within_box_col == 2:
            position_desc += " right"
        position_desc = position_desc.strip()
        
        if position_desc:
            return f"the {position_desc} cell of the {box_descriptions[(box_row, box_col)]} box"
        else:
            return f"the {box_descriptions[(box_row, box_col)]} box"
    
    @staticmethod
    def describe_relative_position(cell1: Tuple[int, int], cell2: Tuple[int, int]) -> str:
        """Describe how one cell relates to another."""
        row1, col1 = cell1
        row2, col2 = cell2
        
        if row1 == row2:
            if abs(col1 - col2) == 1:
                return "right next to"
            elif col1 < col2:
                return "to the left of"
            else:
                return "to the right of"
        elif col1 == col2:
            if abs(row1 - row2) == 1:
                return "directly above/below"
            elif row1 < row2:
                return "above"
            else:
                return "below"
        else:
            return "diagonally from"

class NaturalLanguageExplainer:
    """Generates natural, human-like explanations of solving steps."""
    
    def __init__(self):
        self.spatial_ref = SpatialReference()
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize template phrases for various explanations."""
        self.observation_starters = [
            "I notice that",
            "Looking at",
            "If you observe",
            "Take a look at",
            "You can see that",
            "It's interesting that",
        ]
        
        self.deduction_starters = [
            "This means that",
            "Therefore,",
            "As a result,",
            "This tells us that",
            "We can conclude that",
            "This implies",
        ]
        
        self.pattern_recognition_phrases = [
            "There's a clear pattern where",
            "I can see a pattern forming with",
            "Notice how these cells form a pattern:",
            "An interesting pattern emerges here:",
        ]
    
    def explain_solving_step(self, step: SolveStep, cells: List[List[Cell]]) -> str:
        """Generate a natural explanation for a solving step."""
        if not step:
            return "No valid solving step found."
        
        # Get cell descriptions
        cell_descriptions = [
            f"({cell.row+1}, {cell.col+1})" for cell in step.cells_affected
        ]
        
        # Choose a random starter
        starter = random.choice(self.observation_starters)
        
        # Build explanation based on strategy type
        if step.value_placed is not None:
            explanation = (
                f"{starter} we can place {step.value_placed} in cell "
                f"{cell_descriptions[0]} because {step.explanation}"
            )
        else:
            explanation = (
                f"{starter} {step.explanation} affecting cells "
                f"{', '.join(cell_descriptions)}"
            )
        
        return explanation
    
    def _explain_naked_single(self, step: SolveStep) -> str:
        """Explain a naked single in natural language."""
        cell = step.cells_affected[0]
        value = step.value_placed
        
        templates = [
            f"{random.choice(self.observation_starters)} {self.spatial_ref.describe_location(cell.row, cell.col)} "
            f"can only contain {value}. All other numbers are eliminated by existing values in the row, column, and box.",
            
            f"Looking at {self.spatial_ref.describe_location(cell.row, cell.col)}, "
            f"we can see that {value} is the only possible number that can go here because "
            f"all other numbers are already present in related cells.",
            
            f"This cell in {self.spatial_ref.describe_location(cell.row, cell.col)} "
            f"has only one candidate left: {value}. Everything else has been eliminated."
        ]
        
        return random.choice(templates)
    
    def _explain_hidden_single(self, step: SolveStep) -> str:
        """Explain a hidden single in natural language."""
        cell = step.cells_affected[0]
        value = step.value_placed
        
        templates = [
            f"{random.choice(self.observation_starters)} {value} can only go in "
            f"{self.spatial_ref.describe_location(cell.row, cell.col)} within this unit. "
            f"All other cells in the unit cannot accept {value}.",
            
            f"While this cell has multiple candidates, it's the only cell in its unit that can accept {value}. "
            f"{random.choice(self.deduction_starters)} {value} must go here.",
            
            f"Taking a closer look at {self.spatial_ref.describe_location(cell.row, cell.col)}, "
            f"we can see it's the only place where {value} can go in this unit."
        ]
        
        return random.choice(templates)
    
    def explain_visual_pattern(self, pattern: VisualGroup) -> str:
        """Generate natural language explanation of a visual pattern."""
        # Base explanation
        base = f"{random.choice(self.observation_starters)} a {pattern.pattern_type.name.lower().replace('_', ' ')} "
        
        # Add cell locations
        cells_desc = []
        for row, col in pattern.cells:
            cells_desc.append(f"at position ({row+1},{col+1})")
        
        # Add candidates
        candidates = sorted(pattern.candidates)
        if len(candidates) == 1:
            candidates_desc = f"focusing on number {candidates[0]}"
        else:
            candidates_desc = f"involving numbers {', '.join(map(str, candidates))}"
        
        # Add strength indicator
        if pattern.strength > 0.8:
            strength_desc = "This is a very clear pattern."
        elif pattern.strength > 0.5:
            strength_desc = "This pattern is moderately clear."
        else:
            strength_desc = "This is a subtle pattern."
        
        # Combine descriptions
        explanation = f"{base} {', '.join(cells_desc)} {candidates_desc}. {strength_desc}"
        
        return explanation

    def explain_strategy_selection(self, strategy: Strategy, patterns: List[VisualGroup]) -> str:
        """Explain why a particular strategy was chosen."""
        explanation = f"I've chosen to try {strategy.value} because "
        
        if patterns:
            pattern_desc = self.explain_visual_pattern(patterns[0])
            explanation += f"I noticed {pattern_desc.lower()}"
        else:
            explanation += "it's a good approach for the current grid state"
        
        return explanation
    
    def _explain_aligned_pattern(self, pattern: VisualGroup) -> str:
        """Explain an aligned pattern naturally."""
        cells = pattern.cells
        candidates = pattern.candidates
        
        if len(cells) == 2:
            cell1, cell2 = cells
            return (
                f"{random.choice(self.observation_starters)} two cells "
                f"{self.spatial_ref.describe_relative_position(cell1, cell2)} each other "
                f"share the same candidates: {', '.join(map(str, candidates))}. "
                f"This could be significant!"
            )
        else:
            return (
                f"{random.choice(self.pattern_recognition_phrases)} "
                f"several aligned cells sharing candidates {', '.join(map(str, candidates))}."
            )
    
    def explain_strategy_selection(self, strategy: Strategy, 
                                 patterns: List[VisualGroup]) -> str:
        """Explain why a particular strategy was chosen."""
        relevant_patterns = [p for p in patterns if p.strength > 0.7]
        
        explanation = []
        
        # Explain visual cues first
        if relevant_patterns:
            pattern_desc = self.explain_visual_pattern(relevant_patterns[0])
            explanation.append(pattern_desc)
            explanation.append(f"{random.choice(self.deduction_starters)} "
                            f"trying {strategy.value} might be effective here.")
        else:
            explanation.append(f"Based on the current state of the puzzle, "
                            f"{strategy.value} seems like a good technique to try.")
        
        return " ".join(explanation)
    
    def generate_hint(self, patterns: List[VisualGroup]) -> str:
        """Generate a helpful hint without giving away the solution."""
        if not patterns:
            return "Try looking for cells with few candidates."
        
        pattern = patterns[0]
        return f"Look carefully at the cells {', '.join(f'({r+1},{c+1})' for r,c in pattern.cells)}. " \
               f"There might be an interesting pattern with numbers {', '.join(map(str, pattern.candidates))}."

    def explain_dead_end(self, last_strategy: Strategy) -> str:
        """Explain when a solving attempt reaches a dead end."""
        return (
            f"I tried {last_strategy.value} but couldn't make progress. "
            "We might need to backtrack and try a different approach, "
            "or look for more complex patterns."
        )