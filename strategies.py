# strategies.py

from typing import Set, List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class Strategy(Enum):
    NAKED_SINGLE = "Naked Single"
    HIDDEN_SINGLE = "Hidden Single"
    NAKED_PAIR = "Naked Pair"
    HIDDEN_PAIR = "Hidden Pair"
    NAKED_TRIPLE = "Naked Triple"
    HIDDEN_TRIPLE = "Hidden Triple"
    POINTING_PAIR = "Pointing Pair"
    BOX_LINE_REDUCTION = "Box/Line Reduction"
    XY_WING = "XY-Wing"
    X_WING = "X-Wing"
    SWORDFISH = "Swordfish"

@dataclass
class Cell:
    row: int
    col: int
    value: int
    candidates: Set[int]
    
    def __str__(self):
        return f"Cell({self.row}, {self.col}): {self.value} {self.candidates}"

@dataclass
class SolveStep:
    strategy: Strategy
    cells_affected: List[Cell]
    candidates_removed: Set[int]
    value_placed: Optional[int] = None
    explanation: str = ""

class Grid:
    def __init__(self, puzzle: np.ndarray):
        self.grid = puzzle.copy()
        self.size = 9
        self.cells = [[None for _ in range(9)] for _ in range(9)]
        self.initialize_cells()
        
    def initialize_cells(self):
        """Initialize all cells with their values and candidates."""
        for row in range(self.size):
            for col in range(self.size):
                value = self.grid[row, col]
                candidates = set(range(1, 10)) if value == 0 else set()
                self.cells[row][col] = Cell(row, col, value, candidates)
        
        # Initial candidate computation
        self.compute_all_candidates()
    
    def compute_all_candidates(self):
        """Compute candidates for all empty cells."""
        for row in range(self.size):
            for col in range(self.size):
                if self.cells[row][col].value == 0:
                    self.cells[row][col].candidates = self.compute_candidates(row, col)
    
    def compute_candidates(self, row: int, col: int) -> Set[int]:
        """Compute possible candidates for a specific cell."""
        if self.cells[row][col].value != 0:
            return set()
            
        candidates = set(range(1, 10))
        
        # Remove values from same row
        for c in range(self.size):
            if self.grid[row, c] != 0:
                candidates.discard(self.grid[row, c])
        
        # Remove values from same column
        for r in range(self.size):
            if self.grid[r, col] != 0:
                candidates.discard(self.grid[r, col])
        
        # Remove values from same box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if self.grid[r, c] != 0:
                    candidates.discard(self.grid[r, c])
        
        return candidates
    
    def get_units(self) -> List[List[Cell]]:
        """Get all units (rows, columns, boxes) of the grid."""
        units = []
        
        # Rows
        for row in range(self.size):
            units.append([self.cells[row][col] for col in range(self.size)])
        
        # Columns
        for col in range(self.size):
            units.append([self.cells[row][col] for row in range(self.size)])
        
        # Boxes
        for box_row in range(3):
            for box_col in range(3):
                box = []
                for r in range(box_row*3, box_row*3 + 3):
                    for c in range(box_col*3, box_col*3 + 3):
                        box.append(self.cells[r][c])
                units.append(box)
        
        return units

    def set_value(self, row: int, col: int, value: int):
        """Set a value in a cell and update related candidates."""
        self.grid[row, col] = value
        self.cells[row][col].value = value
        self.cells[row][col].candidates = set()
        
        # Update candidates in same row, column, and box
        for c in range(self.size):
            self.cells[row][c].candidates.discard(value)
        
        for r in range(self.size):
            self.cells[r][col].candidates.discard(value)
        
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                self.cells[r][c].candidates.discard(value)

class HumanSolver:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.solve_steps: List[SolveStep] = []
    
    def find_naked_single(self) -> Optional[SolveStep]:
        """Find a cell with only one candidate."""
        for row in range(9):
            for col in range(9):
                cell = self.grid.cells[row][col]
                if cell.value == 0 and len(cell.candidates) == 1:
                    value = cell.candidates.pop()
                    explanation = f"Cell ({row+1}, {col+1}) has only one candidate: {value}"
                    return SolveStep(
                        strategy=Strategy.NAKED_SINGLE,
                        cells_affected=[cell],
                        candidates_removed=set(),
                        value_placed=value,
                        explanation=explanation
                    )
        return None
    
    def find_hidden_single(self) -> Optional[SolveStep]:
        """Find a value that can only go in one cell in a unit."""
        for unit in self.grid.get_units():
            value_positions: Dict[int, List[Cell]] = {}
            for cell in unit:
                if cell.value == 0:
                    for candidate in cell.candidates:
                        value_positions.setdefault(candidate, []).append(cell)
            
            for value, cells in value_positions.items():
                if len(cells) == 1:
                    cell = cells[0]
                    explanation = f"Value {value} can only go in cell ({cell.row+1}, {cell.col+1}) in this unit"
                    return SolveStep(
                        strategy=Strategy.HIDDEN_SINGLE,
                        cells_affected=[cell],
                        candidates_removed=cell.candidates - {value},
                        value_placed=value,
                        explanation=explanation
                    )
        return None
    
    def find_naked_pair(self) -> Optional[SolveStep]:
        """Find two cells in a unit that share the same two candidates."""
        for unit in self.grid.get_units():
            # Find cells with exactly 2 candidates
            pair_cells = [cell for cell in unit if len(cell.candidates) == 2]
            
            # Check each possible pair
            for i in range(len(pair_cells)):
                for j in range(i + 1, len(pair_cells)):
                    cell1, cell2 = pair_cells[i], pair_cells[j]
                    if cell1.candidates == cell2.candidates:
                        # Found a naked pair
                        candidates = cell1.candidates
                        affected_cells = []
                        candidates_removed = set()
                        
                        # Remove these candidates from other cells in the unit
                        for cell in unit:
                            if cell != cell1 and cell != cell2:
                                before = len(cell.candidates)
                                cell.candidates -= candidates
                                after = len(cell.candidates)
                                if after < before:
                                    affected_cells.append(cell)
                                    candidates_removed.update(candidates & cell.candidates)
                        
                        if affected_cells:
                            explanation = (f"Cells ({cell1.row+1}, {cell1.col+1}) and ({cell2.row+1}, {cell2.col+1}) "
                                        f"share candidates {candidates}")
                            return SolveStep(
                                strategy=Strategy.NAKED_PAIR,
                                cells_affected=affected_cells + [cell1, cell2],
                                candidates_removed=candidates_removed,
                                explanation=explanation
                            )
        return None

    def solve_step(self) -> Optional[SolveStep]:
        """Try to apply each strategy in order of difficulty."""
        strategies = [
            self.find_naked_single,
            self.find_hidden_single,
            self.find_naked_pair,
            # Add more strategies here in order of complexity
        ]
        
        for strategy in strategies:
            if step := strategy():
                # Apply the step
                if step.value_placed is not None:
                    self.grid.set_value(
                        step.cells_affected[0].row,
                        step.cells_affected[0].col,
                        step.value_placed
                    )
                self.solve_steps.append(step)
                return step
        
        return None

    def solve(self) -> Tuple[bool, List[SolveStep]]:
        """
        Solve the puzzle using human strategies.
        Returns (success, steps).
        """
        while True:
            if step := self.solve_step():
                if np.all(self.grid.grid != 0):
                    return True, self.solve_steps
            else:
                break
        
        # If we get here, we couldn't solve it with our current strategies
        return False, self.solve_steps

    def explain_solution(self) -> List[str]:
        """Return a list of human-readable explanations for each solving step."""
        return [f"Step {i+1}: {step.explanation}" for i, step in enumerate(self.solve_steps)]