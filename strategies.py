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
    
    def __hash__(self):
        return hash((self.row, self.col))  # Hash based on position
    
    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        return self.row == other.row and self.col == other.col

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

    def find_hidden_pair(self) -> Optional[SolveStep]:
        """Find a hidden pair in the grid."""
        for unit in self.grid.get_units():
            # Create candidate frequency map
            candidate_cells: Dict[int, List[Cell]] = {}
            for cell in unit:
                if cell.value == 0:
                    for candidate in cell.candidates:
                        if candidate not in candidate_cells:
                            candidate_cells[candidate] = []
                        candidate_cells[candidate].append(cell)
            
            # Look for pairs of numbers that appear in exactly two cells
            candidates = list(candidate_cells.keys())
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    num1, num2 = candidates[i], candidates[j]
                    cells1 = set(candidate_cells[num1])
                    cells2 = set(candidate_cells[num2])
                    
                    # If these numbers appear in exactly the same two cells
                    common_cells = cells1 & cells2
                    if len(common_cells) == 2:
                        cell1, cell2 = common_cells
                        other_candidates_removed = False
                        removed_candidates = set()
                        
                        # Remove all other candidates from these cells
                        for cell in common_cells:
                            before = len(cell.candidates)
                            cell.candidates = {num1, num2}
                            if len(cell.candidates) < before:
                                other_candidates_removed = True
                                removed_candidates.update(cell.candidates - {num1, num2})
                        
                        if other_candidates_removed:
                            explanation = (
                                f"Found hidden pair {num1},{num2} in cells "
                                f"({cell1.row+1},{cell1.col+1}) and ({cell2.row+1},{cell2.col+1})"
                            )
                            return SolveStep(
                                strategy=Strategy.HIDDEN_PAIR,
                                cells_affected=list(common_cells),
                                candidates_removed=removed_candidates,
                                explanation=explanation
                            )
        return None

    def find_pointing_pair(self) -> Optional[SolveStep]:
        """Find pointing pairs in the grid."""
        # Check each 3x3 box
        for box_row in range(3):
            for box_col in range(3):
                # Get cells in this box
                box_cells = []
                for r in range(box_row*3, (box_row+1)*3):
                    for c in range(box_col*3, (box_col+1)*3):
                        if self.grid.cells[r][c].value == 0:
                            box_cells.append(self.grid.cells[r][c])
                
                # Check each candidate
                for num in range(1, 10):
                    cells_with_num = [cell for cell in box_cells 
                                    if num in cell.candidates]
                    
                    if len(cells_with_num) == 2:
                        # Check if they're in same row or column
                        cell1, cell2 = cells_with_num
                        if cell1.row == cell2.row:
                            # Check if we can eliminate from rest of row
                            row = cell1.row
                            affected_cells = []
                            for col in range(9):
                                if col // 3 != box_col:  # Outside the box
                                    cell = self.grid.cells[row][col]
                                    if cell.value == 0 and num in cell.candidates:
                                        cell.candidates.remove(num)
                                        affected_cells.append(cell)
                            
                            if affected_cells:
                                explanation = (
                                    f"Found pointing pair for {num} in row {row+1} "
                                    f"at cells ({cell1.row+1},{cell1.col+1}) "
                                    f"and ({cell2.row+1},{cell2.col+1})"
                                )
                                return SolveStep(
                                    strategy=Strategy.POINTING_PAIR,
                                    cells_affected=[cell1, cell2] + affected_cells,
                                    candidates_removed={num},
                                    explanation=explanation
                                )
                                
                        elif cell1.col == cell2.col:
                            # Check if we can eliminate from rest of column
                            col = cell1.col
                            affected_cells = []
                            for row in range(9):
                                if row // 3 != box_row:  # Outside the box
                                    cell = self.grid.cells[row][col]
                                    if cell.value == 0 and num in cell.candidates:
                                        cell.candidates.remove(num)
                                        affected_cells.append(cell)
                            
                            if affected_cells:
                                explanation = (
                                    f"Found pointing pair for {num} in column {col+1} "
                                    f"at cells ({cell1.row+1},{cell1.col+1}) "
                                    f"and ({cell2.row+1},{cell2.col+1})"
                                )
                                return SolveStep(
                                    strategy=Strategy.POINTING_PAIR,
                                    cells_affected=[cell1, cell2] + affected_cells,
                                    candidates_removed={num},
                                    explanation=explanation
                                )
        return None

    def find_box_line_reduction(self) -> Optional[SolveStep]:
        """Find box-line reductions in the grid."""
        # Check each row
        for row in range(9):
            for num in range(1, 10):
                # Find all cells in this row that have this candidate
                cells_with_num = []
                for col in range(9):
                    cell = self.grid.cells[row][col]
                    if cell.value == 0 and num in cell.candidates:
                        cells_with_num.append(cell)
                
                if cells_with_num and all(c.col // 3 == cells_with_num[0].col // 3 
                                        for c in cells_with_num):
                    # All candidates in same box
                    box_col = cells_with_num[0].col // 3
                    box_row = row // 3
                    affected_cells = []
                    
                    # Remove candidate from other cells in the box
                    for r in range(box_row*3, (box_row+1)*3):
                        if r != row:  # Different row in same box
                            for c in range(box_col*3, (box_col+1)*3):
                                cell = self.grid.cells[r][c]
                                if cell.value == 0 and num in cell.candidates:
                                    cell.candidates.remove(num)
                                    affected_cells.append(cell)
                    
                    if affected_cells:
                        explanation = (
                            f"Found box-line reduction: {num} in row {row+1} "
                            f"appears only in box at column {box_col*3+1}-{box_col*3+3}"
                        )
                        return SolveStep(
                            strategy=Strategy.BOX_LINE_REDUCTION,
                            cells_affected=cells_with_num + affected_cells,
                            candidates_removed={num},
                            explanation=explanation
                        )

        # Check each column (similar to row check)
        for col in range(9):
            for num in range(1, 10):
                cells_with_num = []
                for row in range(9):
                    cell = self.grid.cells[row][col]
                    if cell.value == 0 and num in cell.candidates:
                        cells_with_num.append(cell)
                
                if cells_with_num and all(c.row // 3 == cells_with_num[0].row // 3 
                                        for c in cells_with_num):
                    box_row = cells_with_num[0].row // 3
                    box_col = col // 3
                    affected_cells = []
                    
                    for c in range(box_col*3, (box_col+1)*3):
                        if c != col:
                            for r in range(box_row*3, (box_row+1)*3):
                                cell = self.grid.cells[r][c]
                                if cell.value == 0 and num in cell.candidates:
                                    cell.candidates.remove(num)
                                    affected_cells.append(cell)
                    
                    if affected_cells:
                        explanation = (
                            f"Found box-line reduction: {num} in column {col+1} "
                            f"appears only in box at row {box_row*3+1}-{box_row*3+3}"
                        )
                        return SolveStep(
                            strategy=Strategy.BOX_LINE_REDUCTION,
                            cells_affected=cells_with_num + affected_cells,
                            candidates_removed={num},
                            explanation=explanation
                        )
        return None

    def find_xy_wing(self) -> Optional[SolveStep]:
        """Find XY-Wing patterns."""
        # Find all cells with exactly 2 candidates
        bi_value_cells = []
        for row in range(9):
            for col in range(9):
                cell = self.grid.cells[row][col]
                if cell.value == 0 and len(cell.candidates) == 2:
                    bi_value_cells.append(cell)
        
        # Check each potential pivot
        for pivot in bi_value_cells:
            x, y = pivot.candidates
            
            # Find cells that can see the pivot and have xz or yz candidates
            xz_cells = []
            yz_cells = []
            
            for cell in bi_value_cells:
                if cell != pivot and self._cells_see_each_other(pivot, cell):
                    if x in cell.candidates and len(cell.candidates - {x}) == 1:
                        xz_cells.append(cell)
                    if y in cell.candidates and len(cell.candidates - {y}) == 1:
                        yz_cells.append(cell)
            
            # Check each xz and yz cell pair
            for xz_cell in xz_cells:
                z1 = (xz_cell.candidates - {x}).pop()
                for yz_cell in yz_cells:
                    z2 = (yz_cell.candidates - {y}).pop()
                    
                    if z1 == z2:  # Found XY-Wing
                        z = z1
                        affected_cells = []
                        
                        # Find cells that can see both wing cells
                        for row in range(9):
                            for col in range(9):
                                cell = self.grid.cells[row][col]
                                if (cell.value == 0 and 
                                    cell != pivot and 
                                    cell != xz_cell and 
                                    cell != yz_cell and
                                    z in cell.candidates and
                                    self._cells_see_each_other(cell, xz_cell) and
                                    self._cells_see_each_other(cell, yz_cell)):
                                    cell.candidates.remove(z)
                                    affected_cells.append(cell)
                        
                        if affected_cells:
                            explanation = (
                                f"Found XY-Wing: pivot ({pivot.row+1},{pivot.col+1}) "
                                f"with candidates {x},{y}, connected to "
                                f"({xz_cell.row+1},{xz_cell.col+1}) with {x},{z} and "
                                f"({yz_cell.row+1},{yz_cell.col+1}) with {y},{z}"
                            )
                            return SolveStep(
                                strategy=Strategy.XY_WING,
                                cells_affected=[pivot, xz_cell, yz_cell] + affected_cells,
                                candidates_removed={z},
                                explanation=explanation
                            )
        return None

    def find_x_wing(self) -> Optional[SolveStep]:
        """Find X-Wing patterns in the grid."""
        # Check rows first
        for digit in range(1, 10):
            for row1 in range(9):
                # Find cells in row1 that contain digit as candidate
                row1_cells = []
                for col in range(9):
                    cell = self.grid.cells[row1][col]
                    if cell.value == 0 and digit in cell.candidates:
                        row1_cells.append(cell)
                
                if len(row1_cells) == 2:  # Need exactly 2 cells
                    cols = [cell.col for cell in row1_cells]
                    
                    # Look for matching row
                    for row2 in range(row1 + 1, 9):
                        row2_cells = []
                        for col in cols:
                            cell = self.grid.cells[row2][col]
                            if cell.value == 0 and digit in cell.candidates:
                                row2_cells.append(cell)
                        
                        if len(row2_cells) == 2:  # Found potential X-Wing
                            # Remove digit from other cells in these columns
                            affected_cells = []
                            for col in cols:
                                for row in range(9):
                                    if row != row1 and row != row2:
                                        cell = self.grid.cells[row][col]
                                        if cell.value == 0 and digit in cell.candidates:
                                            cell.candidates.remove(digit)
                                            affected_cells.append(cell)
                            
                            if affected_cells:
                                return SolveStep(
                                    strategy=Strategy.X_WING,
                                    cells_affected=row1_cells + row2_cells + affected_cells,
                                    candidates_removed={digit},
                                    explanation=f"Found X-Wing pattern for {digit} in rows {row1+1},{row2+1} and columns {cols[0]+1},{cols[1]+1}"
                                )
        
        return None

    def find_swordfish(self) -> Optional[SolveStep]:
        """Find Swordfish patterns."""
        for digit in range(1, 10):
            # Check rows first
            # Find rows where digit appears 2-3 times
            valid_rows = []
            row_positions = []  # List of lists of positions for each row
            
            for row in range(9):
                current_positions = []
                for col in range(9):
                    cell = self.grid.cells[row][col]
                    if cell.value == 0 and digit in cell.candidates:
                        current_positions.append((row, col))
                
                if 2 <= len(current_positions) <= 3:
                    valid_rows.append(row)
                    row_positions.append(current_positions)
            
            # Check each combination of three rows
            for i in range(len(valid_rows)):
                for j in range(i + 1, len(valid_rows)):
                    for k in range(j + 1, len(valid_rows)):
                        # Get all columns used in these rows
                        cols_used = set()
                        for pos_list in [row_positions[i], row_positions[j], row_positions[k]]:
                            for _, col in pos_list:
                                cols_used.add(col)
                        
                        # If exactly three columns used, we have a Swordfish
                        if len(cols_used) == 3:
                            cols_list = list(cols_used)
                            pattern_cells = []
                            affected_cells = []
                            
                            # Collect all cells in the pattern
                            for row_idx in [valid_rows[i], valid_rows[j], valid_rows[k]]:
                                for col_idx in cols_list:
                                    cell = self.grid.cells[row_idx][col_idx]
                                    if cell.value == 0 and digit in cell.candidates:
                                        pattern_cells.append(cell)
                            
                            # Remove digit from other cells in affected columns
                            for col_idx in cols_list:
                                for row in range(9):
                                    if row not in [valid_rows[i], valid_rows[j], valid_rows[k]]:
                                        cell = self.grid.cells[row][col_idx]
                                        if cell.value == 0 and digit in cell.candidates:
                                            cell.candidates.remove(digit)
                                            affected_cells.append(cell)
                            
                            if affected_cells:
                                return SolveStep(
                                    strategy=Strategy.SWORDFISH,
                                    cells_affected=pattern_cells + affected_cells,
                                    candidates_removed={digit},
                                    explanation=f"Found Swordfish pattern for digit {digit} in rows "
                                              f"{valid_rows[i]+1}, {valid_rows[j]+1}, {valid_rows[k]+1}"
                                )
        
        return None

    def _cells_see_each_other(self, cell1: Cell, cell2: Cell) -> bool:
        """Check if two cells can see each other (same row, column, or box)."""
        return (cell1.row == cell2.row or 
                cell1.col == cell2.col or 
                (cell1.row // 3 == cell2.row // 3 and 
                 cell1.col // 3 == cell2.col // 3))

    def solve_step(self, specific_strategy: Optional[Strategy] = None) -> Optional[SolveStep]:
        """Try to apply either a specific strategy or all strategies in order of difficulty."""
        if specific_strategy:
            strategy_map = {
                Strategy.NAKED_SINGLE: self.find_naked_single,
                Strategy.HIDDEN_SINGLE: self.find_hidden_single,
                Strategy.NAKED_PAIR: self.find_naked_pair,
                Strategy.HIDDEN_PAIR: self.find_hidden_pair,
                Strategy.POINTING_PAIR: self.find_pointing_pair,
                Strategy.BOX_LINE_REDUCTION: self.find_box_line_reduction,
                Strategy.XY_WING: self.find_xy_wing,
                Strategy.X_WING: self.find_x_wing,
                Strategy.SWORDFISH: self.find_swordfish
            }
            if strategy := strategy_map.get(specific_strategy):
                if step := strategy():
                    if step.value_placed is not None:
                        self.grid.set_value(
                            step.cells_affected[0].row,
                            step.cells_affected[0].col,
                            step.value_placed
                        )
                    self.solve_steps.append(step)
                    return step
        else:
            strategies = [
                self.find_naked_single,
                self.find_hidden_single,
                self.find_naked_pair,
                self.find_hidden_pair,
                self.find_pointing_pair,
                self.find_box_line_reduction,
                self.find_xy_wing,
                self.find_x_wing,
                self.find_swordfish
            ]
            
            for strategy in strategies:
                if step := strategy():
                    if step.value_placed is not None:
                        self.grid.set_value(
                            step.cells_affected[0].row,
                            step.cells_affected[0].col,
                            step.value_placed
                        )
                    self.solve_steps.append(step)
                    return step
        
        return None