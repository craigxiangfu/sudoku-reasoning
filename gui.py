# gui.py

import sys
from typing import List, Set, Optional
import numpy as np
from enum import Enum, auto

# Updated PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QTextEdit,
    QFrame, QGraphicsOpacityEffect, QScrollBar, QScrollArea, QGroupBox, QCheckBox,
    QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPalette, QPainter, QLinearGradient

# Your other imports (strategies, data, solver, etc.) remain unchanged.
from strategies import Strategy
from data import SudokuDataGenerator
from solver import SudokuSolver
from strategies import Strategy, Cell, SolveStep
from strategy_selector import PatternType, VisualPattern
from natural_language_explainer import NaturalLanguageExplainer

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

def keyPressEvent(self, event):
    """Handle keyboard shortcuts."""
    if event.key() == Qt.Key.Key_Space:
        # Space to start/pause solving
        if self.solve_btn.isEnabled():
            self.start_solving()
        else:
            self.pause_solving()
    elif event.key() == Qt.Key.Key_N and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
        # Ctrl+N for new puzzle
        self.generate_new_puzzle()
    elif event.key() == Qt.Key.Key_I and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
        # Ctrl+I for input puzzle
        self.show_input_dialog()
    elif event.key() == Qt.Key.Key_R:
        # R to reset current puzzle
        self.reset_puzzle()

def reset_puzzle(self):
    """Reset the current puzzle to its initial state."""
    if hasattr(self, 'initial_puzzle'):
        self.puzzle = self.initial_puzzle.copy()
        self.solver = SudokuSolver(self.puzzle)
        self.grid.update_grid(self.puzzle, is_original=True)
        self.grid.update_candidates(self.solver.get_all_candidates())
        self.status_bar.update_status("Puzzle reset to initial state", "normal")
        self.explanation_panel.clear()
        self.solve_timer.stop()
        self.solve_btn.setEnabled(True)

def pause_solving(self):
    """Pause the solving process."""
    self.solve_timer.stop()
    self.solve_btn.setEnabled(True)
    self.status_bar.update_status("Solving paused", "normal")
    self.status_bar.stop_solving()

def show_shortcuts_dialog(self):
    """Show available keyboard shortcuts."""
    shortcuts = {
        "Space": "Start/Pause solving",
        "Ctrl+N": "New puzzle",
        "Ctrl+I": "Input puzzle",
        "R": "Reset current puzzle",
        "1-9": "Enter number in selected cell",
        "Delete/Backspace": "Clear selected cell"
    }
    
    msg = QMessageBox(self)
    msg.setWindowTitle("Keyboard Shortcuts")
    msg.setText("Available Shortcuts:")
    msg.setInformativeText("\n".join(f"{key}: {value}" for key, value in shortcuts.items()))
    msg.setIcon(QMessageBox.Icon.Information)
    msg.exec()

def init_ui(self):
    central_widget = QWidget()
    central_widget.setStyleSheet("background-color: #F3F4F6;")
    self.setCentralWidget(central_widget)
    
    # Main layout
    layout = QHBoxLayout(central_widget)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(24)
    
    # Left side: Grid and controls
    left_layout = QVBoxLayout()
    left_layout.setSpacing(16)
    
    # Status bar
    self.status_bar = StatusBar()
    left_layout.addWidget(self.status_bar)
    
    # Controls
    controls = QWidget()
    controls.setStyleSheet("background: white; border-radius: 8px; padding: 12px;")
    controls_layout = QHBoxLayout(controls)
    
    self.difficulty_selector = ModernComboBox()
    self.difficulty_selector.addItems(['easy', 'medium', 'hard'])
    
    self.new_puzzle_btn = ModernButton("New Puzzle", primary=True)
    self.new_puzzle_btn.clicked.connect(self.generate_new_puzzle)
    
    self.input_puzzle_btn = ModernButton("Input Puzzle", primary=False)
    self.input_puzzle_btn.clicked.connect(self.show_input_dialog)
    
    self.solve_btn = ModernButton("Solve Step by Step", primary=False)
    self.solve_btn.clicked.connect(self.start_solving)
    
    for widget in [self.difficulty_selector, self.new_puzzle_btn, 
                  self.input_puzzle_btn, self.solve_btn]:
        controls_layout.addWidget(widget)
    
    left_layout.addWidget(controls)
    
    # Grid with title
    grid_container = QWidget()
    grid_container.setStyleSheet("background: white; border-radius: 8px; padding: 16px;")
    grid_layout = QVBoxLayout(grid_container)
    
    grid_title = QLabel("Sudoku Grid")
    grid_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #111827; margin-bottom: 12px;")
    grid_layout.addWidget(grid_title)
    
    self.grid = ModernSudokuGrid()
    grid_layout.addWidget(self.grid)
    
    left_layout.addWidget(grid_container)
    layout.addLayout(left_layout)
    
    # Right side: Strategy panels
    right_layout = QVBoxLayout()
    right_layout.setSpacing(16)
    
    self.strategy_selector = StrategySelectionPanel()
    self.strategy_stats = StrategyStatsPanel()
    self.explanation_panel = ModernExplanationPanel()
    
    for widget in [self.strategy_selector, self.strategy_stats, self.explanation_panel]:
        right_layout.addWidget(widget)
    
    layout.addLayout(right_layout)
    
    # Setup timer and initialize
    self.solve_timer = QTimer()
    self.solve_timer.timeout.connect(self.solve_one_step)
    self.solve_timer.setInterval(500)
    
    self.generate_new_puzzle()
    self.setGeometry(100, 100, 1400, 900)  # Slightly taller to accommodate new elements
    self.setWindowTitle("Advanced Sudoku Solver")

class StrategyStatsPanel(QWidget):
    """Panel showing strategy success rates and usage statistics."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Strategy Statistics")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #111827;
            padding-bottom: 8px;
        """)
        layout.addWidget(title)
        
        # Stats container
        self.stats_container = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_container)
        self.stats_layout.setSpacing(8)
        
        # Style the container
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 8px;
            }
            .stat-row {
                padding: 8px;
                border-bottom: 1px solid #E5E7EB;
            }
            .stat-name {
                font-weight: 500;
                color: #374151;
            }
            .stat-value {
                color: #2563EB;
            }
        """)
        
        layout.addWidget(self.stats_container)
        self.strategy_stats = {}
    
    def update_stats(self, strategy: str, success: bool):
        """Update statistics for a strategy."""
        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = {"attempts": 0, "successes": 0}
            
        stats = self.strategy_stats[strategy]
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1
        
        self._refresh_display()
    
    def _refresh_display(self):
        """Refresh the statistics display."""
        # Clear current stats
        for i in reversed(range(self.stats_layout.count())):
            self.stats_layout.itemAt(i).widget().setParent(None)
        
        # Add updated stats
        for strategy, stats in self.strategy_stats.items():
            success_rate = (stats["successes"] / stats["attempts"] * 100) if stats["attempts"] > 0 else 0
            
            row = QWidget()
            row_layout = QHBoxLayout(row)
            
            name = QLabel(strategy)
            name.setStyleSheet("font-weight: 500; color: #374151;")
            
            stats_label = QLabel(f"{stats['successes']}/{stats['attempts']} ({success_rate:.1f}%)")
            stats_label.setStyleSheet("color: #2563EB;")
            
            row_layout.addWidget(name)
            row_layout.addWidget(stats_label)
            
            self.stats_layout.addWidget(row)

class StatusBar(QWidget):
    """Modern status bar with progress indicator."""
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Status container with indicator and message
        self.status_container = QWidget()
        self.status_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        status_layout = QHBoxLayout(self.status_container)
        status_layout.setContentsMargins(12, 8, 12, 8)
        
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: #6B7280;
                font-size: 16px;
                margin-right: 8px;
            }
        """)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #111827;
                font-size: 14px;
            }
        """)
        
        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        # Statistics
        self.stats_container = QWidget()
        self.stats_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        stats_layout = QHBoxLayout(self.stats_container)
        stats_layout.setSpacing(16)
        
        # Center: Progress bar
        self.progress_container = QWidget()
        self.progress_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        progress_layout = QVBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(12, 8, 12, 8)
        
        # Progress label
        progress_label_layout = QHBoxLayout()
        self.progress_label = QLabel("Solving Progress")
        self.progress_label.setStyleSheet("""
            QLabel {
                color: #374151;
                font-size: 14px;
                font-weight: 500;
            }
        """)
        self.progress_percentage = QLabel("0%")
        self.progress_percentage.setStyleSheet("""
            QLabel {
                color: #2563EB;
                font-size: 14px;
                font-weight: 500;
            }
        """)
        progress_label_layout.addWidget(self.progress_label)
        progress_label_layout.addWidget(self.progress_percentage)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #F3F4F6;
                border-radius: 4px;
                height: 8px;
                margin: 4px 0;
            }
            QProgressBar::chunk {
                background-color: #2563EB;
                border-radius: 4px;
            }
        """)
        
        progress_layout.addLayout(progress_label_layout)
        progress_layout.addWidget(self.progress_bar)
        
        self.moves_label = QLabel("Moves: 0")
        self.time_label = QLabel("Time: 0:00")
        self.cells_remaining = QLabel("Remaining: 81")
        
        for label in [self.moves_label, self.time_label, self.cells_remaining]:
            label.setStyleSheet("""
                QLabel {
                    color: #374151;
                    font-size: 14px;
                }
            """)
            stats_layout.addWidget(label)
        
        layout.addWidget(self.status_container, stretch=2)
        layout.addWidget(self.progress_container, stretch=1)
        layout.addWidget(self.stats_container, stretch=1)
        
        # Start timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.solving_time = 0
        self.moves_count = 0
        
        # Initialize progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(81)  # Total cells in Sudoku grid
    
    def update_progress(self, filled_cells: int):
        """Update the progress bar based on filled cells."""
        progress = min(filled_cells, 81)
        self.progress_bar.setValue(progress)
        percentage = int((progress / 81) * 100)
        self.progress_percentage.setText(f"{percentage}%")
        self.cells_remaining.setText(f"Remaining: {81 - progress}")
    
    def update_status(self, message: str, state: str = "normal"):
        """Update status with message and state."""
        self.status_label.setText(message)
        
        # Define color schemes for different states
        color_map = {
            "normal": ("#6B7280", "#F9FAFB", "#111827"),  # Indicator, background, text
            "thinking": ("#2563EB", "#EFF6FF", "#1E40AF"),
            "retrying": ("#F59E0B", "#FEF3C7", "#B45309"),
            "success": ("#10B981", "#D1FAE5", "#065F46"),
            "error": ("#EF4444", "#FEE2E2", "#B91C1C")
        }
        
        if state in color_map:
            indicator_color, bg_color, text_color = color_map[state]
            
            # Update indicator color
            self.status_indicator.setStyleSheet(f"""
                QLabel {{
                    color: {indicator_color};
                    font-size: 16px;
                    margin-right: 8px;
                }}
            """)
            
            # Update status label
            self.status_label.setStyleSheet(f"""
                QLabel {{
                    color: {text_color};
                    font-size: 14px;
                    font-weight: {"500" if state != "normal" else "normal"};
                }}
            """)
            
            # Update container background
            self.status_container.setStyleSheet(f"""
                QWidget {{
                    background-color: {bg_color};
                    border-radius: 6px;
                    padding: 8px;
                }}
            """)
    
    def start_solving(self):
        """Start the solving timer."""
        self.timer.start(1000)  # Update every second
        self.solving_time = 0
    
    def stop_solving(self):
        """Stop the solving timer."""
        self.timer.stop()
    
    def update_time(self):
        """Update the solving time display."""
        self.solving_time += 1
        minutes = self.solving_time // 60
        seconds = self.solving_time % 60
        self.time_label.setText(f"Time: {minutes}:{seconds:02d}")
    
    def increment_moves(self):
        """Increment the moves counter."""
        self.moves_count += 1
        self.moves_label.setText(f"Moves: {self.moves_count}")
    
    def update_remaining(self, count: int):
        """Update the remaining cells counter."""
        self.cells_remaining.setText(f"Remaining: {count}")

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
                color: #111827;
            }
            
            QComboBox:hover {
                border-color: #2563EB;
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
            
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #E5E7EB;
                border-radius: 6px;
                selection-background-color: #EFF6FF;
                selection-color: #1E40AF;
                color: #111827;
            }
            
            QComboBox QAbstractItemView::item {
                padding: 8px 12px;
                min-height: 24px;
            }
            
            QComboBox QAbstractItemView::item:hover {
                background-color: #F3F4F6;
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
    def __init__(self, solver=None, explainer=None):
        super().__init__()
        self.solver = solver
        self.explainer = explainer
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Strategy label
        self.strategy_label = QLabel("Strategy")
        self.strategy_label.setStyleSheet("""
            QLabel {
                color: #111827;
                font-size: 14px;
                font-weight: 500;
                padding: 8px;
                background-color: white;
                border-radius: 6px;
            }
        """)
        
        # Pattern visualization
        self.pattern_text = QTextEdit()
        self.pattern_text.setReadOnly(True)
        self.pattern_text.setStyleSheet("""
            QTextEdit {
                color: #111827;
                font-size: 14px;
                background-color: white;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        self.pattern_text.setMinimumHeight(100)
        self.pattern_text.setMaximumHeight(150)
        
        # Explanation text
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setStyleSheet("""
            QTextEdit {
                color: #111827;
                font-size: 14px;
                background-color: white;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        
        layout.addWidget(self.strategy_label)
        layout.addWidget(self.pattern_text)
        layout.addWidget(self.explanation_text)
    
    def clear(self):
        """Clear all text fields."""
        self.strategy_label.setText("")
        self.pattern_text.clear()
        self.explanation_text.clear()
    
    def update_explanation(self, step):
        """Update explanation panel with new step information."""
        # Update strategy label
        self.strategy_label.setText(step.strategy.value)
        
        # Update pattern visualization if available
        if hasattr(step, 'pattern'):
            self.pattern_text.append(
                self.explainer.explain_visual_pattern(step.pattern)
            )
            
        # Update explanation text
        self.explanation_text.append(
            self.explainer.explain_solving_step(step, self.solver.grid.cells)
        )
        
        # Auto-scroll explanation text
        self.explanation_text.verticalScrollBar().setValue(
            self.explanation_text.verticalScrollBar().maximum()
        )

class SudokuSolverGUI(QMainWindow):
    def __init__(self, system):
        super().__init__()
        self.setWindowTitle("Sudoku Reasoning")
        self.system = system
        self.generator = SudokuDataGenerator()
        self.explainer = NaturalLanguageExplainer()
        self.strategy_stats = StrategyStatsPanel()
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
        
        # Add status bar at the top
        self.status_bar = StatusBar()
        left_layout.addWidget(self.status_bar)
        
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
        self.explanation_panel = ModernExplanationPanel(self, self.explainer)
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
    
    def update_status(self, message: str, state: str = "normal"):
        """Update the status bar with a message and state."""
        self.status_bar.update_status(message, state)

    def generate_new_puzzle(self):
        """Generate and display a new puzzle."""
        difficulty = self.difficulty_selector.currentText()
        puzzle, solution = self.generator.generate_puzzle(difficulty)
        self.solver = SudokuSolver(puzzle)
        self.grid.update_grid(puzzle, is_original=True)
        
        # Reset progress
        filled_cells = np.count_nonzero(puzzle)
        self.status_bar.update_progress(filled_cells)
        
        # Clear explanations
        self.explanation_panel.clear()
        self.update_status("Ready to solve", "normal")
    
    def start_solving(self):
        """Start the step-by-step solving process."""
        self.solve_btn.setEnabled(False)
        self.solve_timer.start()
    
    def solve_one_step(self):
        """Perform one solving step using selected strategies."""
        strategies = list(self.strategy_selector.get_selected_strategies())
        
        if not strategies:
            self.update_status("No strategies selected!", "error")
            self.solve_timer.stop()
            return
        
        # Store grid state before attempting strategies
        previous_grid = self.solver.grid.grid.copy()
        current_progress = False
        
        # Try each strategy once
        for strategy in strategies:
            self.update_status(f"Trying {strategy.value}...", "thinking")
            step, current_grid = self.solver.solve_step(strategy)
            
            if step is not None:
                # Progress was made
                current_progress = True
                self.update_status(f"Applied {strategy.value} successfully!", "success")
                
                # Update grid and explanations
                self.grid.update_grid(current_grid)
                self.grid.highlight_strategy_application(step)
                self.explanation_panel.update_explanation(step)
                self.strategy_stats.update_stats(strategy.value, True)
                self.status_bar.increment_moves()
                
                # Update progress
                filled_cells = np.count_nonzero(current_grid)
                self.status_bar.update_progress(filled_cells)
                
                if filled_cells == 81:
                    self.update_status("Puzzle solved!", "success")
                    self.solve_timer.stop()
                
                break
            else:
                self.strategy_stats.update_stats(strategy.value, False)
        
        if not current_progress:
            self.update_status("No progress made with current strategies.", "retrying")
            self.solve_timer.stop()

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