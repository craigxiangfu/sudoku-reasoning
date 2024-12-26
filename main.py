# main.py

import sys
import numpy as np
import argparse
import json
from typing import Optional, Dict
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import QApplication

from data import SudokuDataGenerator
from solver import SudokuSolver, format_grid
from gui import SudokuSolverGUI
from visual_recognition import VisualScanner, ScanPattern
from advanced_learning import AdvancedLearningSystem
from natural_language_explainer import NaturalLanguageExplainer
from strategy_selector import StrategySelector

class SudokuSystem:
    """Central system managing all Sudoku solving components."""
    
    def __init__(self):
        self.generator = SudokuDataGenerator()
        self.visual_scanner = VisualScanner()
        self.learning_system = AdvancedLearningSystem()
        self.strategy_selector = StrategySelector()
        self.explainer = NaturalLanguageExplainer()
        
        # Load learning data
        self.learning_system.load_learned_data()
    
    def solve_puzzle(self, puzzle: np.ndarray, step_by_step: bool = False,
                    scan_pattern: Optional[ScanPattern] = None) -> Dict:
        """
        Solve puzzle with full human-like reasoning.
        Returns dict with solution and analysis data.
        """
        solver = SudokuSolver(puzzle)
        current_grid = puzzle.copy()
        solving_history = []
        
        while True:
            # Visual analysis
            scan_result = self.visual_scanner.scan_grid(
                current_grid, 
                solver.get_all_candidates(),
                pattern=scan_pattern
            )
            
            # Strategy selection based on visual patterns
            strategy = self.strategy_selector.select_next_strategy(
                current_grid,
                solver.available_strategies
            )
            
            # Apply strategy
            start_time = datetime.now()
            step, current_grid = solver.solve_step(strategy)
            time_taken = (datetime.now() - start_time).total_seconds()
            
            if step is None:
                break
                
            # Generate natural explanation
            explanation = self.explainer.explain_solving_step(step, solver.grid.cells)
            
            # Update learning system
            self.learning_system.update_learning(
                current_grid,
                solver.get_all_candidates(),
                strategy.value,
                success=True,
                time_taken=time_taken
            )
            
            solving_history.append({
                'step': step,
                'explanation': explanation,
                'patterns': scan_result.patterns,
                'focus_areas': scan_result.focus_areas,
                'time_taken': time_taken
            })
            
            if step_by_step:
                yield {
                    'grid': current_grid.copy(),
                    'step': solving_history[-1],
                    'scan_result': scan_result
                }
        
        # Get learning state and analysis
        learning_state = self.learning_system.get_learning_state()
        
        return {
            'success': np.all(current_grid != 0),
            'solution': current_grid,
            'history': solving_history,
            'learning_state': learning_state
        }
    
    def analyze_puzzle(self, puzzle: np.ndarray) -> Dict:
        """Analyze puzzle without solving it."""
        scan_result = self.visual_scanner.scan_grid(
            puzzle,
            self.solver.get_all_candidates()
        )
        
        return {
            'patterns': scan_result.patterns,
            'focus_areas': scan_result.focus_areas,
            'difficulty_estimate': self.learning_system.estimate_difficulty(puzzle)
        }

def solve_puzzle_command_line(system: SudokuSystem, puzzle: np.ndarray, 
                            step_by_step: bool = False):
    """Enhanced command-line solving with visual analysis."""
    print("\nOriginal puzzle:")
    print(format_grid(puzzle))
    
    print("\nAnalyzing puzzle...")
    analysis = system.analyze_puzzle(puzzle)
    print(f"Estimated difficulty: {analysis['difficulty_estimate']:.2f}")
    print(f"Detected patterns: {len(analysis['patterns'])}")
    
    if step_by_step:
        print("\nSolving step by step...")
        for state in system.solve_puzzle(puzzle, step_by_step=True):
            print("\nVisual scan result:")
            print(f"Focus areas: {len(state['scan_result'].focus_areas)}")
            print("\nStrategy application:")
            print(state['step']['explanation'])
            print("\nCurrent grid:")
            print(format_grid(state['grid']))
            input("Press Enter for next step...")
    else:
        result = system.solve_puzzle(puzzle, step_by_step=False)
        print("\nSolution found!")
        print("\nKey solving steps:")
        for step in result['history']:
            print(f"\n{step['explanation']}")
        
        print("\nFinal grid:")
        print(format_grid(result['solution']))
        
        print("\nLearning progress:")
        learning_state = result['learning_state']
        print(f"Total patterns learned: {learning_state['total_patterns_learned']}")
        print("Recent discoveries:")
        for discovery in learning_state['recent_discoveries']:
            print(f"- {discovery['type']}: {discovery['success_rate']:.2f} success rate")

def generate_example_puzzle(system: SudokuSystem, difficulty: str) -> np.ndarray:
    """Generate puzzle with difficulty analysis."""
    puzzle, _ = system.generator.generate_puzzle(difficulty)
    analysis = system.analyze_puzzle(puzzle)
    actual_difficulty = analysis['difficulty_estimate']
    
    # Regenerate if difficulty doesn't match target
    while abs(actual_difficulty - {'easy': 0.3, 'medium': 0.6, 'hard': 0.9}[difficulty]) > 0.2:
        puzzle, _ = system.generator.generate_puzzle(difficulty)
        analysis = system.analyze_puzzle(puzzle)
        actual_difficulty = analysis['difficulty_estimate']
    
    return puzzle

def main():
    parser = argparse.ArgumentParser(description='Enhanced Sudoku Solver with Human Reasoning')
    parser.add_argument('--cli', action='store_true', help='Run in command-line mode')
    parser.add_argument('--puzzle', nargs='+', type=int, 
                      help='Input puzzle (81 numbers, row by row, use 0 for empty cells)')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                      default='medium', help='Difficulty for generated puzzle')
    parser.add_argument('--step-by-step', action='store_true',
                      help='Show step-by-step solution in CLI mode')
    parser.add_argument('--scan-pattern', choices=[p.name for p in ScanPattern],
                      help='Visual scanning pattern to use')
    parser.add_argument('--save-learning', action='store_true',
                      help='Save learning progress to file')
    
    args = parser.parse_args()
    
    # Initialize the system
    system = SudokuSystem()
    
    if args.cli:
        # Command-line mode
        if args.puzzle:
            if len(args.puzzle) != 81:
                print("Error: Puzzle must contain exactly 81 numbers")
                return
            puzzle = np.array(args.puzzle).reshape(9, 9)
        else:
            puzzle = generate_example_puzzle(system, args.difficulty)
        
        scan_pattern = ScanPattern[args.scan_pattern] if args.scan_pattern else None
        solve_puzzle_command_line(system, puzzle, args.step_by_step)
        
        if args.save_learning:
            learning_state = system.learning_system.get_learning_state()
            with open('learning_progress.json', 'w') as f:
                json.dump(learning_state, f, indent=2)
    
    else:
        # GUI mode
        app = QApplication(sys.argv)
        window = SudokuSolverGUI(system)
        window.show()
        sys.exit(app.exec())

def solve_puzzle_from_file(system: SudokuSystem, filename: str, 
                          step_by_step: bool = False) -> Optional[np.ndarray]:
    """Enhanced file-based puzzle solving."""
    try:
        puzzle = np.loadtxt(filename, dtype=int)
        if puzzle.shape != (9, 9):
            raise ValueError("Invalid puzzle dimensions")
        
        print(f"\nSolving puzzle from {filename}")
        solve_puzzle_command_line(system, puzzle, step_by_step)
        
        return puzzle
        
    except Exception as e:
        print(f"Error reading puzzle file: {e}")
        return None

if __name__ == "__main__":
    main()