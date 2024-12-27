# solver.py

import numpy as np
from typing import List, Tuple, Optional
from strategies import Grid, HumanSolver, SolveStep, Strategy

class SudokuSolver:
    """
    A human-like Sudoku solver that uses logical strategies and can explain its reasoning.
    """
    
    def __init__(self, puzzle: np.ndarray):
        """Initialize solver with a puzzle."""
        self.original_puzzle = puzzle.copy()
        self.grid = Grid(puzzle)
        self.solver = HumanSolver(self.grid)
        self.available_strategies = [
            Strategy.NAKED_SINGLE,
            Strategy.HIDDEN_SINGLE,
            Strategy.NAKED_PAIR,
            Strategy.HIDDEN_PAIR,
            Strategy.POINTING_PAIR,
            Strategy.BOX_LINE_REDUCTION,
            Strategy.XY_WING,
            Strategy.X_WING,
            Strategy.SWORDFISH
        ]
    
    def solve(self) -> Tuple[bool, np.ndarray, List[str]]:
        """
        Solve the puzzle using human strategies.
        Returns:
            - success: Whether the puzzle was solved
            - solution: The final grid (solved or partial)
            - explanations: List of human-readable solving steps
        """
        success, steps = self.solver.solve()
        explanations = self.solver.explain_solution()
        
        return success, self.grid.grid, explanations
    
    def solve_step(self, strategy: Optional[Strategy] = None) -> Tuple[Optional[SolveStep], np.ndarray]:
        """
        Perform a single solving step using the specified strategy.
        If no strategy is specified, try all available strategies.
        
        Args:
            strategy: Optional specific strategy to use
            
        Returns:
            - step: The solving step taken (None if no step possible)
            - current_grid: The current state of the grid
        """
        if strategy is None:
            # Try all strategies if none specified
            step = self.solver.solve_step()
        else:
            # Try the specific strategy
            if strategy == Strategy.NAKED_SINGLE:
                step = self.solver.find_naked_single()
            elif strategy == Strategy.HIDDEN_SINGLE:
                step = self.solver.find_hidden_single()
            elif strategy == Strategy.NAKED_PAIR:
                step = self.solver.find_naked_pair()
            elif strategy == Strategy.HIDDEN_PAIR:
                step = self.solver.find_hidden_pair()
            elif strategy == Strategy.POINTING_PAIR:
                step = self.solver.find_pointing_pair()
            elif strategy == Strategy.BOX_LINE_REDUCTION:
                step = self.solver.find_box_line_reduction()
            elif strategy == Strategy.XY_WING:
                step = self.solver.find_xy_wing()
            elif strategy == Strategy.X_WING:
                step = self.solver.find_x_wing()
            elif strategy == Strategy.SWORDFISH:
                step = self.solver.find_swordfish()
            else:
                step = None
        
        # If a step was found, apply it
        if step:
            if step.value_placed is not None:
                self.grid.set_value(
                    step.cells_affected[0].row,
                    step.cells_affected[0].col,
                    step.value_placed
                )
        
        return step, self.grid.grid
    
    def get_candidates(self, row: int, col: int) -> set:
        """Get the current candidate numbers for a cell."""
        return self.grid.cells[row][col].candidates
    
    def get_all_candidates(self) -> List[List[set]]:
        """Get candidates for all cells."""
        return [[self.grid.cells[r][c].candidates for c in range(9)] for r in range(9)]

def format_grid(grid: np.ndarray) -> str:
    """Format a grid for pretty printing."""
    result = []
    for i in range(9):
        if i % 3 == 0 and i != 0:
            result.append("-" * 25)
        
        row = []
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row.append("|")
            value = grid[i, j]
            row.append(str(value) if value != 0 else ".")
        
        result.append(" ".join(row))
    return "\n".join(result)

def main():
    # Example usage
    puzzle = np.array([
        [5,3,0,0,7,0,0,0,0],
        [6,0,0,1,9,5,0,0,0],
        [0,9,8,0,0,0,0,6,0],
        [8,0,0,0,6,0,0,0,3],
        [4,0,0,8,0,3,0,0,1],
        [7,0,0,0,2,0,0,0,6],
        [0,6,0,0,0,0,2,8,0],
        [0,0,0,4,1,9,0,0,5],
        [0,0,0,0,8,0,0,7,9]
    ])
    
    print("Original puzzle:")
    print(format_grid(puzzle))
    print("\nSolving...")
    
    solver = SudokuSolver(puzzle)
    success, solution, explanations = solver.solve()
    
    print("\nSolution steps:")
    for explanation in explanations:
        print(explanation)
    
    print("\nFinal grid:")
    print(format_grid(solution))
    
    if success:
        print("\nPuzzle solved successfully!")
    else:
        print("\nCould not completely solve puzzle with current strategies.")
        print("More advanced strategies might be needed.")

if __name__ == "__main__":
    main()