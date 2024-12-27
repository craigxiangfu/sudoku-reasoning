import sys
from typing import List, Set, Optional
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QTextEdit,
    QFrame, QGraphicsOpacityEffect, QScrollArea, QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import QFont, QColor, QPalette, QPainter, QLinearGradient
from enum import Enum, auto
from typing import List, Set, Optional
from strategies import Strategy

from data import SudokuDataGenerator
from solver import SudokuSolver
from strategies import Strategy, Cell, SolveStep
from strategy_selector import PatternType, VisualPattern
from natural_language_explainer import NaturalLanguageExplainer

# -------------------------------------------------------------------------------------------------

class CellHighlight(Enum):
    NONE = auto()
    PATTERN = auto()
    AFFECTED = auto()
    CHANGED = auto()
    RELATED = auto()

class ModernButton(QPushButton):
    def __init__(self, text: str, primary: bool = True):
        super().__init__(text)
        self.primary = primary
        self.setup_style()
    
    def setup_style(self):
        style = """
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                %s
            }
            QPushButton:hover {
                %s
            }
            QPushButton:pressed {
                %s
            }
            QPushButton:disabled {
                background-color: #E5E7EB;
                color: #9CA3AF;
                border: none;
            }
        """
        
        if self.primary:
            style = style % (
                "background-color: #2563EB; color: white; border: none;",
                "background-color: #1D4ED8;",
                "background-color: #1E40AF;"
            )
        else:
            style = style % (
                "background-color: white; color: #2563EB; border: 1px solid #2563EB;",
                "background-color: #EFF6FF;",
                "background-color: #DBEAFE;"
            )
        
        self.setStyleSheet(style)

class ModernComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.setup_style()
    
    def setup_style(self):
        self.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                border: 1px solid #E5E7EB;
                border-radius: 6px;
                background-color: white;
                min-width: 120px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #6B7280;
                margin-right: 8px;
            }
            QComboBox:hover {
                border-color: #2563EB;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #E5E7EB;
                border-radius: 6px;
                background-color: white;
                selection-background-color: #EFF6FF;
                selection-color: #2563EB;
            }
        """)

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
        
        # Value label
        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Use system font with fallback
        self.value_label.setFont(QFont('Helvetica', 20))
        self.layout.addWidget(self.value_label)
        
        # Candidates grid
        self.candidates_widget = QWidget()
        self.candidates_layout = QGridLayout(self.candidates_widget)
        self.candidates_layout.setSpacing(0)
        self.candidates_layout.setContentsMargins(1, 1, 1, 1)
        
        self.candidate_labels = {}
        for i in range(9):
            row, col = divmod(i, 3)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(QFont('Helvetica', 8))
            self.candidates_layout.addWidget(label, row, col)
            self.candidate_labels[i + 1] = label
        
        self.layout.addWidget(self.candidates_widget)
        
        # Set fixed size and style
        self.setFixedSize(64, 64)
        self.setFrameStyle(QFrame.Shape.NoFrame)
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
        if self.value != 0:
            self.value_label.setText(str(self.value))
            self.candidates_widget.hide()
        else:
            self.value_label.setText("")
            self.candidates_widget.show()
            for num, label in self.candidate_labels.items():
                label.setText(str(num) if num in self.candidates else "")
    
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
    
    def update_style(self):
        # Base style with modern aesthetics
        style = """
            QFrame {
                background-color: %s;
                border: none;
                border-radius: 4px;
            }
        """
        
        # Background colors
        if self.highlight == CellHighlight.PATTERN:
            color = "#DBEAFE"  # Light blue
        elif self.highlight == CellHighlight.AFFECTED:
            color = "#FEF3C7"  # Light yellow
        elif self.highlight == CellHighlight.CHANGED:
            color = "#D1FAE5"  # Light green
        elif self.highlight == CellHighlight.RELATED:
            color = "#F3E8FF"  # Light purple
        else:
            color = "#F9FAFB" if self.is_original else "white"
        
        self.setStyleSheet(style % color)
        
        # Text styling
        if self.value != 0:
            color = "#111827" if self.is_original else "#2563EB"
            weight = "600" if self.is_original else "400"
            self.value_label.setStyleSheet(f"""
                color: {color};
                font-weight: {weight};
            """)

class ModernSudokuGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.cells = []
        layout = QGridLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Create cells
        for i in range(9):
            row = []
            for j in range(9):
                cell = SudokuCell()
                layout.addWidget(cell, i, j)
                row.append(cell)
            self.cells.append(row)
        
        # Add box borders
        self.setStyleSheet("""
            QWidget {
                background-color: #E5E7EB;
                border-radius: 8px;
            }
        """)
        
        self.setLayout(layout)
        self.setFixedSize(600, 600)
    
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

class ModernExplanationPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Strategy section
        strategy_layout = QHBoxLayout()
        strategy_label = QLabel("Current Strategy:")
        strategy_label.setStyleSheet("color: #4B5563; font-size: 14px;")
        strategy_layout.addWidget(strategy_label)
        
        self.strategy_label = QLabel()
        self.strategy_label.setStyleSheet("""
            color: #111827;
            font-size: 14px;
            font-weight: 600;
        """)
        strategy_layout.addWidget(self.strategy_label)
        strategy_layout.addStretch()
        layout.addLayout(strategy_layout)
        
        # Create scrollable text areas with proper containers
        patterns_group = QGroupBox("Detected Patterns")
        patterns_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 12px;
                margin-top: 16px;
            }
            QGroupBox::title {
                color: #4B5563;
                font-size: 14px;
                padding: 0 8px;
            }
        """)
        patterns_layout = QVBoxLayout(patterns_group)
        self.pattern_text = QTextEdit()
        self.pattern_text.setReadOnly(True)
        self.pattern_text.setMinimumHeight(150)
        self.pattern_text.setStyleSheet("""
            QTextEdit {
                background-color: #F9FAFB;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        patterns_layout.addWidget(self.pattern_text)
        
        explanations_group = QGroupBox("Solving Process")
        explanations_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 12px;
                margin-top: 16px;
            }
            QGroupBox::title {
                color: #4B5563;
                font-size: 14px;
                padding: 0 8px;
            }
        """)
        explanations_layout = QVBoxLayout(explanations_group)
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMinimumHeight(150)
        self.explanation_text.setStyleSheet("""
            QTextEdit {
                background-color: #F9FAFB;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        explanations_layout.addWidget(self.explanation_text)
        
        layout.addWidget(patterns_group)
        layout.addWidget(explanations_group)
        
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 8px;
            }
        """)
    
class StrategySelectionPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_strategies = set()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Strategy Selection")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #111827;
            padding-bottom: 8px;
        """)
        layout.addWidget(title)
        
        # Strategy groups
        self.create_strategy_group("Basic", [
            Strategy.NAKED_SINGLE,
            Strategy.HIDDEN_SINGLE
        ], layout)
        
        self.create_strategy_group("Intermediate", [
            Strategy.NAKED_PAIR,
            Strategy.HIDDEN_PAIR,
            Strategy.POINTING_PAIR,
            Strategy.BOX_LINE_REDUCTION
        ], layout)
        
        self.create_strategy_group("Advanced", [
            Strategy.XY_WING,
            Strategy.X_WING,
            Strategy.SWORDFISH
        ], layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        select_all = ModernButton("Select All", primary=False)
        select_all.clicked.connect(self.select_all_strategies)
        button_layout.addWidget(select_all)
        
        clear_all = ModernButton("Clear All", primary=False)
        clear_all.clicked.connect(self.clear_all_strategies)
        button_layout.addWidget(clear_all)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        # Apply modern styling
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 8px;
            }
            QGroupBox {
                border: 1px solid #E5E7EB;
                border-radius: 6px;
                margin-top: 1em;
                padding: 8px;
            }
            QGroupBox::title {
                color: #4B5563;
                subcontrol-position: top left;
                padding: 0 3px;
            }
        """)
    
    def create_strategy_group(self, title: str, strategies: List[Strategy], parent_layout: QVBoxLayout):
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        
        for strategy in strategies:
            checkbox = QCheckBox(strategy.value)
            checkbox.setStyleSheet("""
                QCheckBox {
                    color: #374151;
                    font-size: 14px;
                    padding: 4px;
                }
                QCheckBox:hover {
                    background-color: #F3F4F6;
                    border-radius: 4px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border: 2px solid #E5E7EB;
                    border-radius: 4px;
                }
                QCheckBox::indicator:checked {
                    background-color: #2563EB;
                    border-color: #2563EB;
                }
            """)
            checkbox.stateChanged.connect(lambda state, s=strategy: self.on_strategy_toggled(state, s))
            group_layout.addWidget(checkbox)
        
        parent_layout.addWidget(group)
    
    def on_strategy_toggled(self, state: int, strategy: Strategy):
        if state == Qt.CheckState.Checked.value:
            self.selected_strategies.add(strategy)
        else:
            self.selected_strategies.discard(strategy)
    
    def select_all_strategies(self):
        for checkbox in self.findChildren(QCheckBox):
            checkbox.setChecked(True)
    
    def clear_all_strategies(self):
        for checkbox in self.findChildren(QCheckBox):
            checkbox.setChecked(False)
    
    def get_selected_strategies(self) -> Set[Strategy]:
        return self.selected_strategies.copy()

class SudokuSolverGUI(QMainWindow):
    def __init__(self, system):
        super().__init__()
        self.setWindowTitle("Modern Sudoku Solver")
        self.system = system
        self.generator = SudokuDataGenerator()
        self.explainer = NaturalLanguageExplainer()
        self.init_ui()
    
    def init_ui(self):
        # Create central widget with modern styling
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            QWidget {
                background-color: #F3F4F6;
            }
        """)
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)
        
        # Left side: Grid and controls
        left_layout = QVBoxLayout()
        left_layout.setSpacing(16)
        
        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)
        
        self.difficulty_selector = ModernComboBox()
        self.difficulty_selector.addItems(['easy', 'medium', 'hard'])
        controls_layout.addWidget(self.difficulty_selector)
        
        self.new_puzzle_btn = ModernButton("New Puzzle", primary=True)
        self.new_puzzle_btn.clicked.connect(self.generate_new_puzzle)
        controls_layout.addWidget(self.new_puzzle_btn)
        
        self.solve_btn = ModernButton("Solve Step by Step", primary=False)
        self.solve_btn.clicked.connect(self.start_solving)
        controls_layout.addWidget(self.solve_btn)
        
        controls_layout.addStretch()
        left_layout.addLayout(controls_layout)
        
        # Grid
        self.grid = ModernSudokuGrid()
        left_layout.addWidget(self.grid)
        left_layout.addStretch()
        
        layout.addLayout(left_layout)
        
        # Right side: Strategy selector and Explanation panel
        right_layout = QVBoxLayout()
        right_layout.setSpacing(16)
        
        # Strategy selector
        self.strategy_selector = StrategySelectionPanel()
        right_layout.addWidget(self.strategy_selector)
        
        # Explanation panel
        self.explanation_panel = ModernExplanationPanel()
        right_layout.addWidget(self.explanation_panel)
        
        layout.addLayout(right_layout)
        
        # Setup solving animation timer
        self.solve_timer = QTimer()
        self.solve_timer.timeout.connect(self.solve_one_step)
        self.solve_timer.setInterval(500)  # 0.5 seconds between steps
        self.current_strategy_index = 0
        
        # Select default strategies
        self.strategy_selector.select_all_strategies()
        
        self.generate_new_puzzle()
        self.setGeometry(100, 100, 1400, 800)  # Made window wider to accommodate strategy panel
    
    def generate_new_puzzle(self):
        """Generate and display a new puzzle."""
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
        """Start the step-by-step solving process."""
        self.solve_btn.setEnabled(False)
        self.solve_timer.start()
    
    def solve_one_step(self):
        """Perform one solving step using selected strategies."""
        # Get selected strategies
        strategies = list(self.strategy_selector.get_selected_strategies())  # Convert to list for indexing
        if not strategies:
            self.explanation_panel.explanation_text.append(
                "\nPlease select at least one solving strategy."
            )
            self.solve_timer.stop()
            self.solve_btn.setEnabled(True)
            return

        # Try current strategy
        if not hasattr(self, 'current_strategy_index'):
            self.current_strategy_index = 0
        
        if self.current_strategy_index >= len(strategies):
            self.current_strategy_index = 0

        strategy = strategies[self.current_strategy_index]
        step, current_grid = self.solver.solve_step(strategy)
        
        # If current strategy didn't work, try next one
        if step is None:
            self.current_strategy_index += 1
            if self.current_strategy_index < len(strategies):
                # Try next strategy immediately
                self.solve_one_step()
                return
            else:
                # No more strategies to try
                self.current_strategy_index = 0
                self.solve_timer.stop()
                self.explanation_panel.explanation_text.append(
                    "\nNo progress made with current strategies."
                )
                self.solve_btn.setEnabled(True)
                return
        
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
    
    # Set application-wide font
    font = QFont('Helvetica', 10)
    app.setFont(font)
    
    window = SudokuSolverGUI("system")
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()