# data.py

import numpy as np
from typing import Tuple, List, Set, Optional
import random

class SudokuDataGenerator:
    """
    Generates valid Sudoku puzzles with guaranteed unique solutions.
    Includes validation and difficulty rating capabilities.
    """
    
    def __init__(self):
        self.size = 9
        self.box_size = 3
        
    def generate_puzzle(self, difficulty: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a Sudoku puzzle with a unique solution.
        
        Args:
            difficulty: 'easy', 'medium', or 'hard'
            
        Returns:
            Tuple of (puzzle, solution) as numpy arrays
        """
        # Number of cells to remove based on difficulty
        cells_to_remove = {
            'easy': 30,
            'medium': 45,
            'hard': 55
        }.get(difficulty, 45)  # Default to medium if invalid difficulty
        
        solution = self._generate_solved_grid()
        puzzle = self._create_puzzle_from_solution(solution, cells_to_remove)
        
        return puzzle, solution
    
    def is_valid(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """
        Check if a number is valid in a given position.
        
        Args:
            grid: Current Sudoku grid
            row: Row index
            col: Column index
            num: Number to check
            
        Returns:
            Boolean indicating if the number is valid
        """
        # Check row
        if num in grid[row]:
            return False
            
        # Check column
        if num in grid[:, col]:
            return False
            
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if grid[i, j] == num:
                    return False
        
        return True
    
    def get_box_indices(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all indices in the same 3x3 box as (row, col)."""
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        return [(i, j) for i in range(box_row, box_row + 3) 
                      for j in range(box_col, box_col + 3)]
    
    def get_candidates(self, grid: np.ndarray, row: int, col: int) -> Set[int]:
        """Get all valid candidates for a cell."""
        if grid[row, col] != 0:
            return set()
            
        candidates = set(range(1, 10))
        
        # Remove numbers from same row
        candidates -= set(grid[row])
        
        # Remove numbers from same column
        candidates -= set(grid[:, col])
        
        # Remove numbers from same box
        box_indices = self.get_box_indices(row, col)
        for r, c in box_indices:
            candidates.discard(grid[r, c])
            
        return candidates
    
    def _generate_solved_grid(self) -> np.ndarray:
        """Generate a completely solved Sudoku grid."""
        grid = np.zeros((9, 9), dtype=int)
        
        def solve(grid: np.ndarray) -> bool:
            # Find empty cell
            empty = np.where(grid == 0)
            if len(empty[0]) == 0:
                return True
                
            row, col = empty[0][0], empty[1][0]
            
            # Try digits in random order
            for num in random.sample(range(1, 10), 9):
                if self.is_valid(grid, row, col, num):
                    grid[row, col] = num
                    if solve(grid):
                        return True
                    grid[row, col] = 0
            
            return False
        
        solve(grid)
        return grid
    
    def _create_puzzle_from_solution(self, solution: np.ndarray, cells_to_remove: int) -> np.ndarray:
        """
        Create a puzzle by removing numbers from a solved grid,
        ensuring a unique solution remains.
        """
        puzzle = solution.copy()
        positions = list(np.ndindex(9, 9))
        random.shuffle(positions)
        
        removed = 0
        for pos in positions:
            if removed >= cells_to_remove:
                break
                
            # Try removing the number
            backup = puzzle[pos]
            puzzle[pos] = 0
            
            # Check if solution is still unique
            if not self._has_unique_solution(puzzle):
                puzzle[pos] = backup
            else:
                removed += 1
                
        return puzzle
    
    def _has_unique_solution(self, grid: np.ndarray) -> bool:
        """Check if a puzzle has exactly one solution."""
        solutions_found = []
        
        def count_solutions(g: np.ndarray) -> None:
            if len(solutions_found) > 1:
                return
                
            if not np.any(g == 0):
                solutions_found.append(g.copy())
                return
                
            # Find first empty cell
            empty = np.where(g == 0)
            row, col = empty[0][0], empty[1][0]
            
            # Try each possible number
            for num in range(1, 10):
                if self.is_valid(g, row, col, num):
                    g[row, col] = num
                    count_solutions(g)
                    g[row, col] = 0
                    
            return
        
        grid_copy = grid.copy()
        count_solutions(grid_copy)
        return len(solutions_found) == 1
    
    def rate_difficulty(self, puzzle: np.ndarray) -> str:
        """
        Rate the difficulty of a puzzle based on:
        1. Number of empty cells
        2. Distribution of given numbers
        3. Required solving techniques
        
        Returns: 'easy', 'medium', or 'hard'
        """
        # Count empty cells
        empty_count = np.sum(puzzle == 0)
        
        if empty_count < 35:
            return 'easy'
        elif empty_count > 50:
            return 'hard'
            
        # Check distribution of givens in rows/columns/boxes
        row_counts = [np.sum(puzzle[i] != 0) for i in range(9)]
        col_counts = [np.sum(puzzle[:, i] != 0) for i in range(9)]
        
        # Calculate standard deviation of givens distribution
        std_dev = np.std(row_counts + col_counts)
        
        if std_dev < 1.0:  # Even distribution
            return 'easy'
        elif std_dev > 2.0:  # Uneven distribution
            return 'hard'
            
        return 'medium'
    
    def validate_puzzle(self, puzzle: np.ndarray) -> bool:
        """
        Validate that a puzzle follows Sudoku rules:
        - No duplicates in rows
        - No duplicates in columns
        - No duplicates in boxes
        - Has valid dimensions
        """
        if puzzle.shape != (9, 9):
            return False
            
        # Check each row
        for i in range(9):
            row = puzzle[i][puzzle[i] != 0]
            if len(set(row)) != len(row):
                return False
                
        # Check each column
        for j in range(9):
            col = puzzle[:, j][puzzle[:, j] != 0]
            if len(set(col)) != len(col):
                return False
                
        # Check each box
        for box_row in range(3):
            for box_col in range(3):
                box = puzzle[box_row*3:(box_row+1)*3, 
                           box_col*3:(box_col+1)*3]
                box_vals = box[box != 0]
                if len(set(box_vals)) != len(box_vals):
                    return False
                    
        return True

def load_puzzle_from_file(filename: str) -> Optional[np.ndarray]:
    """
    Load a puzzle from a file. File should contain 9 lines with 9 numbers each.
    Use 0 for empty cells.
    """
    try:
        puzzle = np.loadtxt(filename, dtype=int)
        if puzzle.shape != (9, 9):
            raise ValueError("Invalid puzzle dimensions")
        
        generator = SudokuDataGenerator()
        if not generator.validate_puzzle(puzzle):
            raise ValueError("Invalid puzzle: breaks Sudoku rules")
            
        return puzzle
        
    except Exception as e:
        print(f"Error loading puzzle: {e}")
        return None

def save_puzzle_to_file(puzzle: np.ndarray, filename: str) -> bool:
    """Save a puzzle to a file."""
    try:
        np.savetxt(filename, puzzle, fmt='%d')
        return True
    except Exception as e:
        print(f"Error saving puzzle: {e}")
        return False