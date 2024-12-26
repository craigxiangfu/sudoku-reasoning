# test_human_reasoning.py

import numpy as np
import json
from datetime import datetime
from pathlib import Path

from data import SudokuDataGenerator
from visual_recognition import VisualScanner, ScanPattern
from advanced_learning import AdvancedLearningSystem
from strategy_selector import StrategySelector
from natural_language_explainer import NaturalLanguageExplainer
from solver import SudokuSolver

class HumanReasoningTester:
    def __init__(self):
        self.generator = SudokuDataGenerator()
        self.visual_scanner = VisualScanner()
        self.learning_system = AdvancedLearningSystem()
        self.strategy_selector = StrategySelector()
        self.explainer = NaturalLanguageExplainer()
        self.solver = None
    
    def test_visual_recognition(self, puzzle: np.ndarray):
        """Test visual pattern recognition capabilities."""
        print("\n=== Testing Visual Recognition ===")
        
        # Initialize solver to get candidates
        self.solver = SudokuSolver(puzzle)
        candidates = self.solver.get_all_candidates()
        
        # Test different scanning patterns
        for pattern in ScanPattern:
            print(f"\nTesting {pattern.name} scanning pattern:")
            scan_result = self.visual_scanner.scan_grid(puzzle, candidates, pattern=pattern)
            
            print(f"- Eye movements: {len(scan_result.fixation_points)} fixation points")
            print(f"- Detected patterns: {len(scan_result.patterns)}")
            print(f"- Focus areas: {len(scan_result.focus_areas)}")
            
            # Show sample pattern descriptions
            if scan_result.patterns:
                print("\nSample pattern descriptions:")
                for visual_pattern in scan_result.patterns[:2]:
                    # Try both technical and natural language descriptions
                    technical_desc = visual_pattern.get_description()
                    natural_desc = self.explainer.explain_visual_pattern(visual_pattern)
                    print(f"Technical: {technical_desc}")
                    print(f"Natural: {natural_desc}\n")
    
    def test_strategy_selection(self, puzzle: np.ndarray):
        """Test strategy selection process."""
        print("\n=== Testing Strategy Selection ===")
        
        if self.solver is None:
            self.solver = SudokuSolver(puzzle)
        
        candidates = self.solver.get_all_candidates()
        scan_result = self.visual_scanner.scan_grid(puzzle, candidates)
        
        # Test strategy selection over multiple steps
        for step_num in range(3):  # Test first 3 moves
            print(f"\nStep {step_num + 1}:")
            strategy = self.strategy_selector.select_next_strategy(
                puzzle, self.solver.available_strategies
            )
            
            print(f"Selected Strategy: {strategy.value}")
            explanation = self.explainer.explain_strategy_selection(
                strategy, scan_result.patterns
            )
            print(f"Reasoning: {explanation}")
            
            # Apply strategy and update puzzle state
            step, new_puzzle = self.solver.solve_step(strategy)
            if step is None:
                print("No more steps possible")
                break
            
            puzzle = new_puzzle
            candidates = self.solver.get_all_candidates()
            scan_result = self.visual_scanner.scan_grid(puzzle, candidates)
            
            # Update learning
            self.learning_system.update_learning(
                puzzle,
                self.solver.get_all_candidates(),
                strategy.value,
                success=True,
                time_taken=1.0
            )
    
    def test_learning_system(self):
        """Test learning and adaptation capabilities."""
        print("\n=== Testing Learning System ===")
        
        # Generate and solve multiple puzzles
        difficulties = ['easy', 'medium', 'hard']
        for difficulty in difficulties:
            print(f"\nTesting {difficulty} puzzle:")
            puzzle, solution = self.generator.generate_puzzle(difficulty)
            
            # Solve puzzle and gather learning data
            solver = SudokuSolver(puzzle)
            start_time = datetime.now()
            success, final_grid, explanations = solver.solve()
            time_taken = (datetime.now() - start_time).total_seconds()
            
            # Update learning system
            self.learning_system.update_learning(
                puzzle,
                solver.get_all_candidates(),
                'test_strategy',
                success=success,
                time_taken=time_taken
            )
        
        # Show learning progress
        learning_state = self.learning_system.get_learning_state()
        print("\nLearning Progress:")
        print(json.dumps(learning_state, indent=2))
    
    def test_natural_explanations(self, puzzle: np.ndarray):
        """Test natural language explanations."""
        print("\n=== Testing Natural Explanations ===")
        
        if self.solver is None:
            self.solver = SudokuSolver(puzzle)
        
        candidates = self.solver.get_all_candidates()
        scan_result = self.visual_scanner.scan_grid(puzzle, candidates)
        
        # Test pattern explanations
        if scan_result.patterns:
            print("\nPattern Explanations:")
            for pattern in scan_result.patterns[:2]:
                explanation = self.explainer.explain_visual_pattern(pattern)
                print(f"- {explanation}")
        
        # Test solving step explanations
        print("\nSolving Step Explanations:")
        for step_num in range(2):  # Test first 2 steps
            step, current_grid = self.solver.solve_step()
            if step is None:
                print("No more steps possible")
                break
            explanation = self.explainer.explain_solving_step(step, self.solver.grid.cells)
            print(f"Step {step_num + 1}: {explanation}")
    
    def run_full_test(self):
        """Run a complete test of all human reasoning components."""
        print("Starting Human Reasoning System Test")
        print("===================================")
        
        # Generate test puzzle
        puzzle, solution = self.generator.generate_puzzle('medium')
        print("\nTest puzzle generated:")
        print(puzzle)
        
        # Run component tests
        self.test_visual_recognition(puzzle)
        self.test_strategy_selection(puzzle)
        self.test_learning_system()
        self.test_natural_explanations(puzzle)
        
        print("\nTest completed!")

def main():
    tester = HumanReasoningTester()
    tester.run_full_test()

if __name__ == "__main__":
    main()