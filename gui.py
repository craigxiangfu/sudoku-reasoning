# gui.py

import sys
from typing import List, Set, Optional
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QTextEdit,
    QFrame, QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPalette
from enum import Enum, auto

from data import SudokuDataGenerator
from solver import SudokuSolver
from strategies import Strategy, Cell, SolveStep
from strategy_selector import PatternType, VisualPattern
from natural_language_explainer import NaturalLanguageExplainer

class CellHighlight(Enum):
    NONE = auto()
    PATTERN = auto()
    AFFECTED = auto()
    CHANGED = auto()
    RELATED = auto()

class SudokuCell(QFrame):
    def __init__(self):
        super().__init__()
        self.value = 0
        self.candidates = set()
        self.is_original = False
        self.highlight = CellHighlight.NONE
        
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        
        # Value label (large font for actual values)
        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setFont(QFont('Arial', 20))
        self.layout.addWidget(self.value_label)
        
        # Candidates grid (3x3 small numbers)
        self.candidates_widget = QWidget()
        self.candidates_layout = QGridLayout(self.candidates_widget)
        self.candidates_layout.setSpacing(0)
        self.candidates_layout.setContentsMargins(1, 1, 1, 1)
        
        self.candidate_labels = {}
        for i in range(9):
            row, col = divmod(i, 3)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(QFont('Arial', 8))
            self.candidates_layout.addWidget(label, row, col)
            self.candidate_labels[i + 1] = label
        
        self.layout.addWidget(self.candidates_widget)
        
        # Set fixed size and style
        self.setFixedSize(60, 60)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        self.update_style()
        
    def update_value(self, value: int, is_original: bool = False):
        self.value = value
        self.is_original = is_original
        if value != 0:
            self.candidates = set()
        self.update_display()
        self.update_style()
    
    def update_candidates(self, candidates: Set[int]):
        if self.value == 0:
            self.candidates = candidates
            self.update_display()
    
    def update_display(self):
        # Update main value
        if self.value != 0:
            self.value_label.setText(str(self.value))
            self.candidates_widget.hide()
        else:
            self.value_label.setText("")
            self.candidates_widget.show()
            # Update candidate labels
            for num, label in self.candidate_labels.items():
                label.setText(str(num) if num in self.candidates else "")
    
    def update_style(self):
        # Base style
        style = """
            QFrame {
                border: 1px solid gray;
                background-color: %s;
            }
        """
        
        # Determine background color based on highlight state
        if self.highlight == CellHighlight.PATTERN:
            color = "#e3f2fd"  # Light blue for pattern cells
        elif self.highlight == CellHighlight.AFFECTED:
            color = "#fff3e0"  # Light orange for affected cells
        elif self.highlight == CellHighlight.CHANGED:
            color = "#e8f5e9"  # Light green for changed cells
        elif self.highlight == CellHighlight.RELATED:
            color = "#f3e5f5"  # Light purple for related cells
        else:
            color = "#f5f5f5" if self.is_original else "white"
        
        self.setStyleSheet(style % color)
        
        # Update font color and weight
        if self.value != 0:
            self.value_label.setStyleSheet("""
                color: %s;
                font-weight: %s;
            """ % (
                "black" if self.is_original else "#1976d2",
                "bold" if self.is_original else "normal"
            ))

    def animate_highlight(self, highlight_type: CellHighlight):
        """Animate cell highlighting with fade effect."""
        self.highlight = highlight_type
        
        # Create fade effect
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        
        # Create animation
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(500)  # 500ms duration
        animation.setStartValue(0.5)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Update style and start animation
        self.update_style()
        animation.start()

class SudokuGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.cells = []
        layout = QGridLayout()
        layout.setSpacing(0)
        
        # Create 9x9 grid of cells
        for i in range(9):
            row = []
            for j in range(9):
                cell = SudokuCell()
                layout.addWidget(cell, i, j)
                row.append(cell)
            self.cells.append(row)
        
        # Add thicker borders for 3x3 boxes
        for i in range(9):
            for j in range(9):
                cell = self.cells[i][j]
                style = cell.styleSheet()[:-1]  # Remove last brace
                if i % 3 == 0 and i != 0:
                    style += "border-top: 2px solid black;"
                if j % 3 == 0 and j != 0:
                    style += "border-left: 2px solid black;"
                style += "}"
                cell.setStyleSheet(style)
        
        self.setLayout(layout)
    
    def update_grid(self, puzzle: np.ndarray, is_original: bool = False):
        """Update grid values."""
        for i in range(9):
            for j in range(9):
                self.cells[i][j].update_value(puzzle[i][j], is_original)
    
    def update_candidates(self, candidates: List[List[Set[int]]]):
        """Update candidate numbers for all cells."""
        for i in range(9):
            for j in range(9):
                self.cells[i][j].update_candidates(candidates[i][j])
    
    def highlight_pattern(self, pattern: VisualPattern):
        """Highlight cells involved in a pattern."""
        for row, col in pattern.cells:
            self.cells[row][col].animate_highlight(CellHighlight.PATTERN)
    
    def highlight_strategy_application(self, step: SolveStep):
        """Highlight cells affected by a strategy application."""
        # Reset all highlights first
        self.clear_highlights()
        
        # Highlight main affected cells
        for cell in step.cells_affected:
            self.cells[cell.row][cell.col].animate_highlight(CellHighlight.AFFECTED)
        
        # If value was placed, highlight that cell specially
        if step.value_placed is not None:
            self.cells[step.cells_affected[0].row][step.cells_affected[0].col].animate_highlight(
                CellHighlight.CHANGED
            )
    
    def clear_highlights(self):
        """Clear all cell highlights."""
        for row in self.cells:
            for cell in row:
                cell.highlight = CellHighlight.NONE
                cell.update_style()

class ExplanationPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Strategy section
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Current Strategy:"))
        self.strategy_label = QLabel()
        self.strategy_label.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        strategy_layout.addWidget(self.strategy_label)
        strategy_layout.addStretch()
        layout.addLayout(strategy_layout)
        
        # Pattern section
        layout.addWidget(QLabel("Detected Patterns:"))
        self.pattern_text = QTextEdit()
        self.pattern_text.setReadOnly(True)
        self.pattern_text.setMaximumHeight(100)
        layout.addWidget(self.pattern_text)
        
        # Explanation section
        layout.addWidget(QLabel("Solving Process:"))
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        layout.addWidget(self.explanation_text)

class SudokuSolverGUI(QMainWindow):
    def __init__(self, system):
        super().__init__()
        self.setWindowTitle("Sudoku Solver with Pattern Visualization")
        self.system = system
        self.generator = SudokuDataGenerator()
        self.explainer = NaturalLanguageExplainer()
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Left side: Grid and controls
        left_layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.difficulty_selector = QComboBox()
        self.difficulty_selector.addItems(['easy', 'medium', 'hard'])
        controls_layout.addWidget(QLabel("Difficulty:"))
        controls_layout.addWidget(self.difficulty_selector)
        
        self.new_puzzle_btn = QPushButton("New Puzzle")
        self.new_puzzle_btn.clicked.connect(self.generate_new_puzzle)
        controls_layout.addWidget(self.new_puzzle_btn)
        
        self.solve_btn = QPushButton("Solve Step by Step")
        self.solve_btn.clicked.connect(self.start_solving)
        controls_layout.addWidget(self.solve_btn)
        
        left_layout.addLayout(controls_layout)
        
        # Grid
        self.grid = SudokuGrid()
        left_layout.addWidget(self.grid)
        
        layout.addLayout(left_layout)
        
        # Right side: Explanation panel
        self.explanation_panel = ExplanationPanel()
        layout.addWidget(self.explanation_panel)
        
        # Setup solving animation timer
        self.solve_timer = QTimer()
        self.solve_timer.timeout.connect(self.solve_one_step)
        self.solve_timer.setInterval(1500)  # 1.5 seconds between steps
        
        self.generate_new_puzzle()
        self.setGeometry(100, 100, 1200, 700)
    
    def generate_new_puzzle(self):
        difficulty = self.difficulty_selector.currentText()
        self.puzzle, self.solution = self.generator.generate_puzzle(difficulty)
        self.solver = SudokuSolver(self.puzzle)
        
        # Show initial state
        self.grid.update_grid(self.puzzle, is_original=True)
        self.grid.update_candidates(self.solver.get_all_candidates())
        
        # Clear explanations
        self.explanation_panel.strategy_label.setText("")
        self.explanation_panel.pattern_text.clear()
        self.explanation_panel.explanation_text.clear()
        
        self.solve_timer.stop()
        self.solve_btn.setEnabled(True)
        self.new_puzzle_btn.setEnabled(True)
    
    def start_solving(self):
        self.solve_btn.setEnabled(False)
        self.solve_timer.start()
    
    def solve_one_step(self):
        # Get next solving step
        step, current_grid = self.solver.solve_step()
        
        if step is None:
            self.solve_timer.stop()
            if np.all(current_grid != 0):
                self.explanation_panel.explanation_text.append("\nPuzzle solved successfully!")
            else:
                self.explanation_panel.explanation_text.append(
                    "\nCould not solve further with current strategies."
                )
            self.solve_btn.setEnabled(True)
            return
        
        # Update grid and candidates
        self.grid.update_grid(current_grid)
        self.grid.update_candidates(self.solver.get_all_candidates())
        
        # Show pattern if available
        if hasattr(step, 'pattern'):
            self.grid.highlight_pattern(step.pattern)
            self.explanation_panel.pattern_text.append(
                self.explainer.explain_visual_pattern(step.pattern)
            )
        
        # Highlight strategy application
        self.grid.highlight_strategy_application(step)
        
        # Update explanation panel
        self.explanation_panel.strategy_label.setText(step.strategy.value)
        self.explanation_panel.explanation_text.append(
            self.explainer.explain_solving_step(step, self.solver.grid.cells)
        )
        
        # Auto-scroll explanation text
        self.explanation_panel.explanation_text.verticalScrollBar().setValue(
            self.explanation_panel.explanation_text.verticalScrollBar().maximum()
        )

def main():
    app = QApplication(sys.argv)
    window = SudokuSolverGUI("system")
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()