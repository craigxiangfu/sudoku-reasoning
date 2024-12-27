# strategy_selector.py

from enum import Enum, auto
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import math

from strategies import Strategy, Grid, Cell

class PatternType(Enum):
    ALIGNED_PAIR = auto()       # Two cells aligned in row/column
    BOX_PATTERN = auto()        # Pattern within a 3x3 box
    LINE_INTERACTION = auto()   # Interaction between line and box
    WING_PATTERN = auto()       # XY-Wing like patterns
    CHAIN_PATTERN = auto()      # Chain of related candidates
    FISH_PATTERN = auto()       # X-Wing/Swordfish like patterns

@dataclass
class VisualPattern:
    pattern_type: PatternType
    cells: List[Tuple[int, int]]
    candidates: Set[int]
    strength: float  # How visually obvious the pattern is (0.0 to 1.0)
    description: str

class StrategyDifficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXTREME = 4

@dataclass
class StrategyProfile:
    strategy: Strategy
    difficulty: StrategyDifficulty
    preferred_patterns: List[PatternType]
    min_candidates: int  # Minimum candidates needed in grid
    success_rate: float = 0.5  # Initial 50% success rate

class StrategySelector:
    def __init__(self):
        self.strategy_profiles = self._initialize_strategy_profiles()
        self.success_history: Dict[Strategy, List[bool]] = defaultdict(list)
        self.pattern_history: Dict[PatternType, int] = defaultdict(int)
        self.last_successful_strategy: Optional[Strategy] = None
        
    def _initialize_strategy_profiles(self) -> Dict[Strategy, StrategyProfile]:
        """Initialize strategy profiles with their characteristics."""
        return {
            Strategy.NAKED_SINGLE: StrategyProfile(
                strategy=Strategy.NAKED_SINGLE,
                difficulty=StrategyDifficulty.EASY,
                preferred_patterns=[],  # No patterns needed
                min_candidates=1
            ),
            Strategy.HIDDEN_SINGLE: StrategyProfile(
                strategy=Strategy.HIDDEN_SINGLE,
                difficulty=StrategyDifficulty.EASY,
                preferred_patterns=[PatternType.LINE_INTERACTION],
                min_candidates=2
            ),
            Strategy.NAKED_PAIR: StrategyProfile(
                strategy=Strategy.NAKED_PAIR,
                difficulty=StrategyDifficulty.MEDIUM,
                preferred_patterns=[PatternType.ALIGNED_PAIR],
                min_candidates=4
            ),
            Strategy.HIDDEN_PAIR: StrategyProfile(
                strategy=Strategy.HIDDEN_PAIR,
                difficulty=StrategyDifficulty.MEDIUM,
                preferred_patterns=[PatternType.BOX_PATTERN],
                min_candidates=4
            ),
            Strategy.POINTING_PAIR: StrategyProfile(
                strategy=Strategy.POINTING_PAIR,
                difficulty=StrategyDifficulty.MEDIUM,
                preferred_patterns=[PatternType.LINE_INTERACTION],
                min_candidates=4
            ),
            Strategy.BOX_LINE_REDUCTION: StrategyProfile(
                strategy=Strategy.BOX_LINE_REDUCTION,
                difficulty=StrategyDifficulty.MEDIUM,
                preferred_patterns=[PatternType.LINE_INTERACTION],
                min_candidates=4
            ),
            Strategy.XY_WING: StrategyProfile(
                strategy=Strategy.XY_WING,
                difficulty=StrategyDifficulty.HARD,
                preferred_patterns=[PatternType.WING_PATTERN],
                min_candidates=6
            ),
            Strategy.X_WING: StrategyProfile(
                strategy=Strategy.X_WING,
                difficulty=StrategyDifficulty.HARD,
                preferred_patterns=[PatternType.FISH_PATTERN],
                min_candidates=8
            ),
            Strategy.SWORDFISH: StrategyProfile(
                strategy=Strategy.SWORDFISH,
                difficulty=StrategyDifficulty.EXTREME,
                preferred_patterns=[PatternType.FISH_PATTERN],
                min_candidates=12
            )
        }

    def detect_visual_patterns(self, grid: Grid) -> List[VisualPattern]:
        """Detect visual patterns in the grid using human-like scanning."""
        patterns = []
        patterns.extend(self._scan_for_aligned_pairs(grid))
        patterns.extend(self._scan_for_box_patterns(grid))
        patterns.extend(self._scan_for_line_interactions(grid))
        patterns.extend(self._scan_for_wing_patterns(grid))
        patterns.extend(self._scan_for_fish_patterns(grid))
        return patterns

    def _scan_for_aligned_pairs(self, grid: Grid) -> List[VisualPattern]:
        """Scan for aligned pairs in rows and columns."""
        patterns = []
        
        # Scan rows
        for row in range(9):
            for col1 in range(9):
                for col2 in range(col1 + 1, 9):
                    cell1 = grid.cells[row][col1]
                    cell2 = grid.cells[row][col2]
                    if (cell1.value == 0 and cell2.value == 0 and 
                        len(cell1.candidates & cell2.candidates) >= 2):
                        strength = 1.0 if col2 == col1 + 1 else 0.7  # Adjacent cells more obvious
                        patterns.append(VisualPattern(
                            pattern_type=PatternType.ALIGNED_PAIR,
                            cells=[(row, col1), (row, col2)],
                            candidates=cell1.candidates & cell2.candidates,
                            strength=strength,
                            description=f"Aligned pair in row {row+1}"
                        ))

        # Scan columns (similar to rows)
        for col in range(9):
            for row1 in range(9):
                for row2 in range(row1 + 1, 9):
                    cell1 = grid.cells[row1][col]
                    cell2 = grid.cells[row2][col]
                    if (cell1.value == 0 and cell2.value == 0 and 
                        len(cell1.candidates & cell2.candidates) >= 2):
                        strength = 1.0 if row2 == row1 + 1 else 0.7
                        patterns.append(VisualPattern(
                            pattern_type=PatternType.ALIGNED_PAIR,
                            cells=[(row1, col), (row2, col)],
                            candidates=cell1.candidates & cell2.candidates,
                            strength=strength,
                            description=f"Aligned pair in column {col+1}"
                        ))
        
        return patterns

    def _scan_for_box_patterns(self, grid: Grid) -> List[VisualPattern]:
        """Scan for patterns within 3x3 boxes."""
        patterns = []
        
        for box_row in range(3):
            for box_col in range(3):
                # Get all cells in this box
                box_cells = []
                box_candidates = defaultdict(list)
                
                for r in range(box_row*3, (box_row+1)*3):
                    for c in range(box_col*3, (box_col+1)*3):
                        cell = grid.cells[r][c]
                        if cell.value == 0:
                            box_cells.append((r, c))
                            for candidate in cell.candidates:
                                box_candidates[candidate].append((r, c))
                
                # Look for patterns in candidate distributions
                for candidate, cells in box_candidates.items():
                    if len(cells) == 2 or len(cells) == 3:
                        # Check if cells are aligned
                        rows = {r for r, _ in cells}
                        cols = {c for _, c in cells}
                        
                        strength = 0.8 if len(rows) == 1 or len(cols) == 1 else 0.6
                        patterns.append(VisualPattern(
                            pattern_type=PatternType.BOX_PATTERN,
                            cells=cells,
                            candidates={candidate},
                            strength=strength,
                            description=f"Box pattern for {candidate} in box ({box_row+1},{box_col+1})"
                        ))
        
        return patterns

    def _scan_for_line_interactions(self, grid: Grid) -> List[VisualPattern]:
        """Scan for interactions between lines (rows/columns) and boxes."""
        patterns = []
        
        # Scan for row-box interactions
        for row in range(9):
            box_row = row // 3
            for candidate in range(1, 10):
                # Find cells in this row with this candidate
                cells = []
                for col in range(9):
                    cell = grid.cells[row][col]
                    if cell.value == 0 and candidate in cell.candidates:
                        cells.append((row, col))
                
                if 2 <= len(cells) <= 3:
                    # Check if all cells are in the same box
                    box_cols = {col // 3 for _, col in cells}
                    if len(box_cols) == 1:
                        patterns.append(VisualPattern(
                            pattern_type=PatternType.LINE_INTERACTION,
                            cells=cells,
                            candidates={candidate},
                            strength=0.9,
                            description=f"Row-box interaction for {candidate} in row {row+1}"
                        ))
        
        # Similar scan for column-box interactions
        for col in range(9):
            box_col = col // 3
            for candidate in range(1, 10):
                cells = []
                for row in range(9):
                    cell = grid.cells[row][col]
                    if cell.value == 0 and candidate in cell.candidates:
                        cells.append((row, col))
                
                if 2 <= len(cells) <= 3:
                    box_rows = {row // 3 for row, _ in cells}
                    if len(box_rows) == 1:
                        patterns.append(VisualPattern(
                            pattern_type=PatternType.LINE_INTERACTION,
                            cells=cells,
                            candidates={candidate},
                            strength=0.9,
                            description=f"Column-box interaction for {candidate} in column {col+1}"
                        ))
        
        return patterns

    def _scan_for_wing_patterns(self, grid: Grid) -> List[VisualPattern]:
        """Scan for XY-Wing like patterns."""
        patterns = []
        
        # Find all cells with exactly 2 candidates
        bi_value_cells = []
        for row in range(9):
            for col in range(9):
                cell = grid.cells[row][col]
                if cell.value == 0 and len(cell.candidates) == 2:
                    bi_value_cells.append((row, col))
        
        # Look for potential wing patterns
        for pivot_pos in bi_value_cells:
            pivot = grid.cells[pivot_pos[0]][pivot_pos[1]]
            x, y = pivot.candidates
            
            # Look for connected bi-value cells
            for wing1_pos in bi_value_cells:
                if wing1_pos == pivot_pos:
                    continue
                    
                wing1 = grid.cells[wing1_pos[0]][wing1_pos[1]]
                if x in wing1.candidates or y in wing1.candidates:
                    for wing2_pos in bi_value_cells:
                        if wing2_pos in (pivot_pos, wing1_pos):
                            continue
                            
                        wing2 = grid.cells[wing2_pos[0]][wing2_pos[1]]
                        if (x in wing2.candidates or y in wing2.candidates) and wing1.candidates & wing2.candidates:
                            patterns.append(VisualPattern(
                                pattern_type=PatternType.WING_PATTERN,
                                cells=[pivot_pos, wing1_pos, wing2_pos],
                                candidates=pivot.candidates | wing1.candidates | wing2.candidates,
                                strength=0.7,
                                description=f"Potential wing pattern with pivot at ({pivot_pos[0]+1},{pivot_pos[1]+1})"
                            ))
        
        return patterns

    def _scan_for_fish_patterns(self, grid: Grid) -> List[VisualPattern]:
        """Scan for X-Wing and Swordfish patterns."""
        patterns = []
        
        def scan_dimension(by_rows: bool):
            for candidate in range(1, 10):
                # Find rows/columns where candidate appears 2-3 times
                candidate_positions = defaultdict(list)
                
                for i in range(9):
                    positions = []
                    for j in range(9):
                        row = i if by_rows else j
                        col = j if by_rows else i
                        cell = grid.cells[row][col]
                        if cell.value == 0 and candidate in cell.candidates:
                            positions.append((row, col))
                    
                    if 2 <= len(positions) <= 3:
                        candidate_positions[i] = positions
                
                # Look for X-Wing patterns (2x2)
                if len(candidate_positions) >= 2:
                    for i1, pos1 in candidate_positions.items():
                        for i2, pos2 in candidate_positions.items():
                            if i1 < i2 and len(pos1) == len(pos2) == 2:
                                cols1 = {c for _, c in pos1}
                                cols2 = {c for _, c in pos2}
                                if cols1 == cols2:
                                    pattern_cells = pos1 + pos2
                                    patterns.append(VisualPattern(
                                        pattern_type=PatternType.FISH_PATTERN,
                                        cells=pattern_cells,
                                        candidates={candidate},
                                        strength=0.8,
                                        description=f"Potential X-Wing pattern for {candidate}"
                                    ))
        
        # Scan both rows and columns
        scan_dimension(True)
        scan_dimension(False)
        
        return patterns

    def select_next_strategy(self, grid: Grid, 
                           available_strategies: List[Strategy]) -> Strategy:
        """
        Select the most appropriate strategy based on visual patterns,
        previous success, and grid state.
        """
        # Detect patterns in current grid
        patterns = self.detect_visual_patterns(grid)
        
        # Count total candidates in grid
        total_candidates = sum(
            len(cell.candidates) 
            for row in grid.cells 
            for cell in row 
            if cell.value == 0
        )
        
        # Score each strategy
        strategy_scores = {}
        for strategy in available_strategies:
            profile = self.strategy_profiles[strategy]
            
            # Base score from success rate
            score = profile.success_rate * 2.0
            
# Pattern matching score
            pattern_score = 0.0
            for pattern in patterns:
                if pattern.pattern_type in profile.preferred_patterns:
                    pattern_score += pattern.strength
            
            # Normalize pattern score
            if profile.preferred_patterns:
                pattern_score /= len(profile.preferred_patterns)
            score += pattern_score * 3.0  # Pattern matching is important
            
            # Candidate count appropriateness
            if total_candidates >= profile.min_candidates:
                score += 1.0
            else:
                score -= 2.0  # Penalize if not enough candidates
            
            # Difficulty progression
            if self.last_successful_strategy:
                last_profile = self.strategy_profiles[self.last_successful_strategy]
                # Prefer strategies of similar or slightly higher difficulty
                diff_delta = profile.difficulty.value - last_profile.difficulty.value
                if diff_delta == 0:
                    score += 0.5  # Same difficulty
                elif diff_delta == 1:
                    score += 0.3  # Slightly harder
                elif diff_delta > 1:
                    score -= 0.5  # Much harder
                else:
                    score -= 0.2  # Easier
            
            # Recent success history (last 5 attempts)
            recent_history = self.success_history[strategy][-5:]
            if recent_history:
                recent_success_rate = sum(1 for x in recent_history if x) / len(recent_history)
                score += recent_success_rate
            
            # Pattern history bonus
            for pattern in patterns:
                if pattern.pattern_type in profile.preferred_patterns:
                    pattern_frequency = self.pattern_history[pattern.pattern_type]
                    if pattern_frequency > 0:
                        score += 0.2  # Bonus for familiar pattern types
            
            strategy_scores[strategy] = score
        
        # Select strategy with highest score
        selected_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        return selected_strategy

    def update_strategy_success(self, strategy: Strategy, success: bool):
        """Update success history and rates for a strategy."""
        # Update success history
        self.success_history[strategy].append(success)
        recent_history = self.success_history[strategy][-10:]  # Last 10 attempts
        
        # Update success rate with exponential moving average
        if recent_history:
            alpha = 0.2  # Learning rate
            new_rate = sum(1 for x in recent_history if x) / len(recent_history)
            current_rate = self.strategy_profiles[strategy].success_rate
            self.strategy_profiles[strategy].success_rate = (
                (1 - alpha) * current_rate + alpha * new_rate
            )
        
        if success:
            self.last_successful_strategy = strategy

    def update_pattern_history(self, patterns: List[VisualPattern]):
        """Update pattern frequency history."""
        for pattern in patterns:
            self.pattern_history[pattern.pattern_type] += 1

    def get_strategy_explanation(self, strategy: Strategy, 
                               patterns: List[VisualPattern]) -> str:
        """Generate a human-like explanation for why a strategy was chosen."""
        profile = self.strategy_profiles[strategy]
        
        # Find relevant patterns
        relevant_patterns = [p for p in patterns 
                           if p.pattern_type in profile.preferred_patterns]
        
        explanation = f"Choosing {strategy.value} because: "
        reasons = []
        
        if relevant_patterns:
            pattern_desc = ", ".join(p.description for p in relevant_patterns[:2])
            reasons.append(f"I noticed {pattern_desc}")
        
        if self.last_successful_strategy:
            last_profile = self.strategy_profiles[self.last_successful_strategy]
            if profile.difficulty == last_profile.difficulty:
                reasons.append("it's a similar difficulty to the last successful strategy")
            elif profile.difficulty.value > last_profile.difficulty.value:
                reasons.append("we might need a more advanced technique")
        
        recent_success = self.success_history[strategy][-5:]
        if recent_success and sum(recent_success) / len(recent_success) > 0.7:
            reasons.append("it has worked well recently")
        
        if not reasons:
            reasons.append("it seems appropriate for the current grid state")
        
        return explanation + " and ".join(reasons)