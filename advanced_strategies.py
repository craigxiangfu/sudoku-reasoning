# advanced_strategies.py

from typing import List, Set, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
from strategies import Strategy, Cell, SolveStep, Grid
from strategy_selector import VisualPattern, PatternType

class AdvancedStrategies:
    """Implementation of advanced human solving strategies."""
    
    @staticmethod
    def find_hidden_pair(grid: Grid) -> Optional[SolveStep]:
        """
        Find hidden pairs in the grid.
        A hidden pair occurs when two cells in a unit share two candidates
        that don't appear elsewhere in the unit.
        """
        for unit in grid.get_units():
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
                        
                        # Remove all other candidates from these cells
                        for cell in common_cells:
                            before = len(cell.candidates)
                            cell.candidates = {num1, num2}
                            if len(cell.candidates) < before:
                                other_candidates_removed = True
                        
                        if other_candidates_removed:
                            explanation = (
                                f"Found hidden pair {num1},{num2} in cells "
                                f"({cell1.row+1},{cell1.col+1}) and ({cell2.row+1},{cell2.col+1})"
                            )
                            return SolveStep(
                                strategy=Strategy.HIDDEN_PAIR,
                                cells_affected=list(common_cells),
                                candidates_removed=set(),
                                explanation=explanation
                            )
        return None

    @staticmethod
    def find_hidden_triple(grid: Grid) -> Optional[SolveStep]:
        """
        Find hidden triples in the grid.
        A hidden triple occurs when three cells share three candidates
        that don't appear elsewhere in the unit.
        """
        for unit in grid.get_units():
            # Create candidate frequency map
            candidate_cells: Dict[int, List[Cell]] = {}
            for cell in unit:
                if cell.value == 0:
                    for candidate in cell.candidates:
                        if candidate not in candidate_cells:
                            candidate_cells[candidate] = []
                        candidate_cells[candidate].append(cell)
            
            # Look for triples of numbers that appear in exactly three cells
            candidates = list(candidate_cells.keys())
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    for k in range(j + 1, len(candidates)):
                        nums = {candidates[i], candidates[j], candidates[k]}
                        cells = set()
                        for num in nums:
                            cells.update(candidate_cells[num])
                        
                        if len(cells) == 3:
                            # Found a hidden triple
                            other_candidates_removed = False
                            for cell in cells:
                                before = len(cell.candidates)
                                cell.candidates &= nums
                                if len(cell.candidates) < before:
                                    other_candidates_removed = True
                            
                            if other_candidates_removed:
                                cell_list = list(cells)
                                explanation = (
                                    f"Found hidden triple {nums} in cells "
                                    f"({cell_list[0].row+1},{cell_list[0].col+1}), "
                                    f"({cell_list[1].row+1},{cell_list[1].col+1}), and "
                                    f"({cell_list[2].row+1},{cell_list[2].col+1})"
                                )
                                return SolveStep(
                                    strategy=Strategy.HIDDEN_TRIPLE,
                                    cells_affected=cell_list,
                                    candidates_removed=set(),
                                    explanation=explanation
                                )
        return None

    @staticmethod
    def find_pointing_pair(grid: Grid) -> Optional[SolveStep]:
        """
        Find pointing pairs in the grid.
        A pointing pair occurs when a candidate appears only in two cells
        in a box, and these cells share a row or column.
        """
        # Check each 3x3 box
        for box_row in range(3):
            for box_col in range(3):
                # Get cells in this box
                box_cells = []
                for r in range(box_row*3, (box_row+1)*3):
                    for c in range(box_col*3, (box_col+1)*3):
                        if grid.cells[r][c].value == 0:
                            box_cells.append(grid.cells[r][c])
                
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
                            eliminated = False
                            for col in range(9):
                                if col // 3 != box_col:  # Outside the box
                                    cell = grid.cells[row][col]
                                    if cell.value == 0 and num in cell.candidates:
                                        cell.candidates.remove(num)
                                        eliminated = True
                            
                            if eliminated:
                                explanation = (
                                    f"Found pointing pair for {num} in row {row+1} "
                                    f"at cells ({cell1.row+1},{cell1.col+1}) "
                                    f"and ({cell2.row+1},{cell2.col+1})"
                                )
                                return SolveStep(
                                    strategy=Strategy.POINTING_PAIR,
                                    cells_affected=[cell1, cell2],
                                    candidates_removed={num},
                                    explanation=explanation
                                )
                                
                        elif cell1.col == cell2.col:
                            # Check if we can eliminate from rest of column
                            col = cell1.col
                            eliminated = False
                            for row in range(9):
                                if row // 3 != box_row:  # Outside the box
                                    cell = grid.cells[row][col]
                                    if cell.value == 0 and num in cell.candidates:
                                        cell.candidates.remove(num)
                                        eliminated = True
                            
                            if eliminated:
                                explanation = (
                                    f"Found pointing pair for {num} in column {col+1} "
                                    f"at cells ({cell1.row+1},{cell1.col+1}) "
                                    f"and ({cell2.row+1},{cell2.col+1})"
                                )
                                return SolveStep(
                                    strategy=Strategy.POINTING_PAIR,
                                    cells_affected=[cell1, cell2],
                                    candidates_removed={num},
                                    explanation=explanation
                                )
        return None

    @staticmethod
    def find_box_line_reduction(grid: Grid) -> Optional[SolveStep]:
        """
        Find box-line reductions.
        This occurs when a candidate in a row/column appears only in one box,
        allowing elimination of that candidate from the rest of the box.
        """
        # Check each row
        for row in range(9):
            for num in range(1, 10):
                # Find all cells in this row that have this candidate
                cells_with_num = []
                for col in range(9):
                    cell = grid.cells[row][col]
                    if cell.value == 0 and num in cell.candidates:
                        cells_with_num.append(cell)
                
                if cells_with_num and all(c.col // 3 == cells_with_num[0].col // 3 
                                        for c in cells_with_num):
                    # All candidates in same box
                    box_col = cells_with_num[0].col // 3
                    box_row = row // 3
                    eliminated = False
                    
                    # Remove candidate from other cells in the box
                    for r in range(box_row*3, (box_row+1)*3):
                        if r != row:  # Different row in same box
                            for c in range(box_col*3, (box_col+1)*3):
                                cell = grid.cells[r][c]
                                if cell.value == 0 and num in cell.candidates:
                                    cell.candidates.remove(num)
                                    eliminated = True
                    
                    if eliminated:
                        explanation = (
                            f"Found box-line reduction: {num} in row {row+1} "
                            f"appears only in box at column {box_col*3+1}-{box_col*3+3}"
                        )
                        return SolveStep(
                            strategy=Strategy.BOX_LINE_REDUCTION,
                            cells_affected=cells_with_num,
                            candidates_removed={num},
                            explanation=explanation
                        )

        # Check each column (similar to row check)
        for col in range(9):
            for num in range(1, 10):
                cells_with_num = []
                for row in range(9):
                    cell = grid.cells[row][col]
                    if cell.value == 0 and num in cell.candidates:
                        cells_with_num.append(cell)
                
                if cells_with_num and all(c.row // 3 == cells_with_num[0].row // 3 
                                        for c in cells_with_num):
                    box_row = cells_with_num[0].row // 3
                    box_col = col // 3
                    eliminated = False
                    
                    for c in range(box_col*3, (box_col+1)*3):
                        if c != col:
                            for r in range(box_row*3, (box_row+1)*3):
                                cell = grid.cells[r][c]
                                if cell.value == 0 and num in cell.candidates:
                                    cell.candidates.remove(num)
                                    eliminated = True
                    
                    if eliminated:
                        explanation = (
                            f"Found box-line reduction: {num} in column {col+1} "
                            f"appears only in box at row {box_row*3+1}-{box_row*3+3}"
                        )
                        return SolveStep(
                            strategy=Strategy.BOX_LINE_REDUCTION,
                            cells_affected=cells_with_num,
                            candidates_removed={num},
                            explanation=explanation
                        )
        return None

    @staticmethod
    def find_xy_wing(grid: Grid) -> Optional[SolveStep]:
        """
        Find XY-Wing patterns.
        This involves three cells: a pivot with two candidates (xy)
        and two cells sharing one candidate each with the pivot (xz, yz).
        """
        # Find all cells with exactly 2 candidates
        pivot_candidates = []
        for row in range(9):
            for col in range(9):
                cell = grid.cells[row][col]
                if cell.value == 0 and len(cell.candidates) == 2:
                    pivot_candidates.append(cell)
        
        # Check each potential pivot
        for pivot in pivot_candidates:
            x, y = pivot.candidates
            
            # Find cells that can see the pivot and have xz or yz candidates
            xz_cells = []
            yz_cells = []
            
            for cell in pivot_candidates:
                if cell != pivot and (
                    cell.row == pivot.row or 
                    cell.col == pivot.col or 
                    (cell.row//3 == pivot.row//3 and cell.col//3 == pivot.col//3)
                ):
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
                        # Find cells that can see both xz and yz cells
                        eliminated = False
                        affected_cells = []
                        
                        for row in range(9):
                            for col in range(9):
                                cell = grid.cells[row][col]
                                if (cell.value == 0 and 
                                    cell != pivot and 
                                    cell != xz_cell and 
                                    cell != yz_cell and
                                    z in cell.candidates):
                                    
                                    # Check if cell can see both wing cells
                                    sees_xz = (
                                        cell.row == xz_cell.row or
                                        cell.col == xz_cell.col or
                                        (cell.row//3 == xz_cell.row//3 and 
                                         cell.col//3 == xz_cell.col//3)
                                    )
                                    sees_yz = (
                                        cell.row == yz_cell.row or
                                        cell.col == yz_cell.col or
                                        (cell.row//3 == yz_cell.row//3 and 
                                         cell.col//3 == yz_cell.col//3)
                                    )
                                    
                                    if sees_xz and sees_yz:
                                        cell.candidates.remove(z)
                                        eliminated = True
                                        affected_cells.append(cell)
                        
                        if eliminated:
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

    @staticmethod
    def find_swordfish(grid: Grid) -> Optional[SolveStep]:
        """
        Find Swordfish patterns.
        A Swordfish occurs when a candidate appears in 2-3 cells in each of three different rows,
        and these cells are confined to three columns (or vice versa).
        """
        def find_swordfish_in_dimension(grid: Grid, by_rows: bool) -> Optional[SolveStep]:
            size = 9
            eliminated = False
            affected_cells = []
            explanation = ""
            
            for digit in range(1, 10):
                # Find rows/cols where digit appears 2-3 times
                valid_lines = []
                positions = []  # List of lists of positions for each line
                
                for i in range(size):
                    current_positions = []
                    for j in range(size):
                        row = i if by_rows else j
                        col = j if by_rows else i
                        cell = grid.cells[row][col]
                        if cell.value == 0 and digit in cell.candidates:
                            current_positions.append((row, col))
                    
                    if 2 <= len(current_positions) <= 3:
                        valid_lines.append(i)
                        positions.append(current_positions)
                
                # Check each combination of three lines
                for i in range(len(valid_lines)):
                    for j in range(i + 1, len(valid_lines)):
                        for k in range(j + 1, len(valid_lines)):
                            # Get all columns used in these rows
                            cols_used = set()
                            for pos_list in [positions[i], positions[j], positions[k]]:
                                for row, col in pos_list:
                                    cols_used.add(col if by_rows else row)
                            
                            # If exactly three columns used, we have a Swordfish
                            if len(cols_used) == 3:
                                cols_list = list(cols_used)
                                pattern_cells = []
                                
                                # Collect all cells in the pattern
                                for row_idx in [valid_lines[i], valid_lines[j], valid_lines[k]]:
                                    for col_idx in cols_list:
                                        row = row_idx if by_rows else col_idx
                                        col = col_idx if by_rows else row_idx
                                        if grid.cells[row][col].value == 0 and digit in grid.cells[row][col].candidates:
                                            pattern_cells.append(grid.cells[row][col])
                                
                                # Remove digit from other cells in affected columns/rows
                                for col_idx in cols_list:
                                    for row in range(size):
                                        if by_rows and row not in [valid_lines[i], valid_lines[j], valid_lines[k]]:
                                            cell = grid.cells[row][col_idx]
                                            if cell.value == 0 and digit in cell.candidates:
                                                cell.candidates.remove(digit)
                                                eliminated = True
                                                affected_cells.append(cell)
                                        elif not by_rows and col_idx not in [valid_lines[i], valid_lines[j], valid_lines[k]]:
                                            cell = grid.cells[col_idx][row]
                                            if cell.value == 0 and digit in cell.candidates:
                                                cell.candidates.remove(digit)
                                                eliminated = True
                                                affected_cells.append(cell)
                                
                                if eliminated:
                                    dimension = "rows" if by_rows else "columns"
                                    explanation = (
                                        f"Found Swordfish pattern for digit {digit} in {dimension} "
                                        f"{valid_lines[i]+1}, {valid_lines[j]+1}, {valid_lines[k]+1} "
                                        f"and columns {', '.join(str(c+1) for c in cols_list)}"
                                    )
                                    return SolveStep(
                                        strategy=Strategy.SWORDFISH,
                                        cells_affected=pattern_cells + affected_cells,
                                        candidates_removed={digit},
                                        explanation=explanation
                                    )
            return None
        
        # Try finding Swordfish by rows, then by columns
        return (find_swordfish_in_dimension(grid, True) or 
                find_swordfish_in_dimension(grid, False))

    @staticmethod
    def find_singles_chain(grid: Grid) -> Optional[SolveStep]:
        """
        Find chains of strongly linked candidates (where two cells must contain
        opposite values of a candidate) to eliminate possibilities.
        """
        # Implementation of singles chains
        # This is an advanced technique that humans use to solve difficult puzzles
        pass  # TODO: Implement singles chains strategy

    @staticmethod
    def find_xy_chain(grid: Grid) -> Optional[SolveStep]:
        """
        Find XY-Chains where bivalue cells are linked together to make eliminations.
        """
        # Implementation of XY-chains
        # This is another advanced technique that humans use
        pass  # TODO: Implement XY-chains strategy