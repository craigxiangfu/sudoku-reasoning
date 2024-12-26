# visual_recognition.py

from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from collections import defaultdict
import random

from visual_patterns import VisualGroup, PatternType

class ScanPattern(Enum):
    ROW_WISE = auto()      # Left to right, top to bottom
    COLUMN_WISE = auto()   # Top to bottom, left to right
    BOX_WISE = auto()      # Within each 3x3 box
    SPIRAL = auto()        # Outside to inside
    RANDOM_WALK = auto()   # Simulated natural eye movement

@dataclass
class VisualGroup:
    """A group of cells that form a visual pattern."""
    cells: List[Tuple[int, int]]
    strength: float  # How visually obvious the pattern is (0-1)
    pattern_type: str
    candidates: Set[int]

@dataclass
class ScanResult:
    """Result of a visual scan, including fixation points and patterns."""
    fixation_points: List[Tuple[int, int]]  # Where the "eye" focused
    scan_order: List[Tuple[int, int]]       # Order of cell examination
    patterns: List[VisualGroup]             # Detected patterns
    focus_areas: List[Tuple[int, int, float]]  # Areas of high interest (row, col, intensity)

class VisualScanner:
    """Simulates human-like visual scanning of the Sudoku grid."""
    
    def __init__(self):
        self.last_fixation = None
        self.attention_map = np.ones((9, 9)) # Attention weight for each cell
        self.pattern_memory = defaultdict(float)  # Long-term pattern memory
        
    def scan_grid(self, grid: np.ndarray, candidates: List[List[Set[int]]], 
                 pattern: ScanPattern = None) -> ScanResult:
        """
        Perform a human-like visual scan of the grid.
        """
        if pattern is None:
            pattern = random.choice(list(ScanPattern))
        
        fixation_points = []
        scan_order = []
        patterns = []
        
        if pattern == ScanPattern.ROW_WISE:
            scan_order = self._row_wise_scan()
        elif pattern == ScanPattern.COLUMN_WISE:
            scan_order = self._column_wise_scan()
        elif pattern == ScanPattern.BOX_WISE:
            scan_order = self._box_wise_scan()
        elif pattern == ScanPattern.SPIRAL:
            scan_order = self._spiral_scan()
        else:  # RANDOM_WALK
            scan_order = self._random_walk_scan()

        # Simulate eye fixations
        fixation_points = self._simulate_eye_movement(scan_order)
        
        # Detect patterns along scan path
        patterns = self._detect_patterns_along_path(grid, candidates, scan_order)
        
        # Calculate focus areas based on pattern density and novelty
        focus_areas = self._calculate_focus_areas(patterns)
        
        # Update attention map based on findings
        self._update_attention_map(patterns, focus_areas)
        
        return ScanResult(fixation_points, scan_order, patterns, focus_areas)
    
    def _simulate_eye_movement(self, scan_order: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Simulate natural eye movement with fixations and saccades."""
        fixations = []
        current_position = (4, 4)  # Start at center
        
        for target in scan_order:
            # Calculate distance to target
            distance = ((target[0] - current_position[0])**2 + 
                       (target[1] - current_position[1])**2)**0.5
            
            if distance > 2:  # If target is far, add intermediate fixations
                steps = max(1, int(distance / 2))
                for i in range(steps):
                    row = int(current_position[0] + (target[0] - current_position[0]) * (i+1) / steps)
                    col = int(current_position[1] + (target[1] - current_position[1]) * (i+1) / steps)
                    fixations.append((row, col))
            
            fixations.append(target)
            current_position = target
            
            # Add small random variations to simulate natural eye movement
            if random.random() < 0.2:  # 20% chance of slight deviation
                row = min(8, max(0, target[0] + random.randint(-1, 1)))
                col = min(8, max(0, target[1] + random.randint(-1, 1)))
                fixations.append((row, col))
        
        return fixations
    
    def _detect_patterns_along_path(self, grid: np.ndarray, 
                                  candidates: List[List[Set[int]]], 
                                  scan_order: List[Tuple[int, int]]) -> List[VisualGroup]:
        """Detect patterns as they would be noticed during visual scanning."""
        patterns = []
        recent_cells = []  # Last few cells examined
        
        for cell in scan_order:
            recent_cells.append(cell)
            if len(recent_cells) > 4:  # Keep a moving window of cells
                recent_cells.pop(0)
            
            # Check for patterns in recent cells
            row, col = cell
            current_candidates = candidates[row][col]
            
            # Look for aligned patterns
            aligned = self._find_aligned_cells(recent_cells, candidates)
            if aligned:
                patterns.append(aligned)
            
            # Look for candidate patterns
            candidate_pattern = self._find_candidate_patterns(recent_cells, candidates)
            if candidate_pattern:
                patterns.append(candidate_pattern)
            
            # Update pattern memory
            self._update_pattern_memory(patterns)
        
        return patterns
    
    def _find_aligned_cells(self, cells: List[Tuple[int, int]], 
                          candidates: List[List[Set[int]]]) -> Optional[VisualGroup]:
        """Find cells that are aligned and share candidates."""
        if len(cells) < 2:
            return None
        
        # Check for cells in same row
        row_cells = [c for c in cells if c[0] == cells[-1][0]]
        if len(row_cells) >= 2:
            shared_candidates = set.intersection(*[candidates[r][c] for r, c in row_cells])
            if shared_candidates:
                return VisualGroup(
                    cells=row_cells,
                    strength=0.8 if len(row_cells) > 2 else 0.6,
                    pattern_type="aligned_row",
                    candidates=shared_candidates
                )
        
        # Check for cells in same column
        col_cells = [c for c in cells if c[1] == cells[-1][1]]
        if len(col_cells) >= 2:
            shared_candidates = set.intersection(*[candidates[r][c] for r, c in col_cells])
            if shared_candidates:
                return VisualGroup(
                    cells=col_cells,
                    strength=0.8 if len(col_cells) > 2 else 0.6,
                    pattern_type="aligned_column",
                    candidates=shared_candidates
                )
        
        return None
    
    def _find_candidate_patterns(self, cells: List[Tuple[int, int]], 
                               candidates: List[List[Set[int]]]) -> Optional[VisualGroup]:
        """Find patterns in candidate distributions."""
        if len(cells) < 2:
            return None
        
        # Get candidates for recent cells
        cell_candidates = {cell: candidates[cell[0]][cell[1]] for cell in cells}
        
        # Look for cells sharing exactly the same candidates
        for size in [2, 3]:  # Look for pairs and triples
            for i in range(len(cells) - size + 1):
                group = cells[i:i+size]
                shared_candidates = set.intersection(*[cell_candidates[c] for c in group])
                if len(shared_candidates) == size:
                    return VisualGroup(
                        cells=group,
                        strength=0.9 if size == 2 else 0.7,
                        pattern_type=f"naked_{size}",
                        candidates=shared_candidates
                    )
        
        return None
    
    def _update_pattern_memory(self, patterns: List[VisualGroup]):
        """Update long-term memory of pattern frequencies and effectiveness."""
        for pattern in patterns:
            key = (pattern.pattern_type, len(pattern.cells), len(pattern.candidates))
            self.pattern_memory[key] += 1
    
    def _calculate_focus_areas(self, patterns: List[VisualGroup]) -> List[Tuple[int, int, float]]:
        """Calculate areas that deserve more attention based on pattern density."""
        focus_map = np.zeros((9, 9))
        
        for pattern in patterns:
            for row, col in pattern.cells:
                focus_map[row, col] += pattern.strength
        
        # Normalize and find high-focus areas
        focus_map = focus_map / (np.max(focus_map) + 1e-6)
        focus_areas = []
        
        for row in range(9):
            for col in range(9):
                if focus_map[row, col] > 0.5:  # High focus threshold
                    focus_areas.append((row, col, focus_map[row, col]))
        
        return focus_areas
    
    def _update_attention_map(self, patterns: List[VisualGroup], 
                            focus_areas: List[Tuple[int, int, float]]):
        """Update attention weights based on discovered patterns."""
        decay_factor = 0.95  # Gradual decay of attention
        self.attention_map *= decay_factor
        
        # Increase attention for areas with patterns
        for pattern in patterns:
            for row, col in pattern.cells:
                self.attention_map[row, col] += pattern.strength
        
        # Increase attention for focus areas
        for row, col, strength in focus_areas:
            self.attention_map[row, col] += strength
        
        # Normalize
        self.attention_map /= np.max(self.attention_map)
    
    # Different scanning patterns
    def _row_wise_scan(self) -> List[Tuple[int, int]]:
        """Left to right, top to bottom scan."""
        return [(r, c) for r in range(9) for c in range(9)]
    
    def _column_wise_scan(self) -> List[Tuple[int, int]]:
        """Top to bottom, left to right scan."""
        return [(r, c) for c in range(9) for r in range(9)]
    
    def _box_wise_scan(self) -> List[Tuple[int, int]]:
        """Scan each 3x3 box."""
        scan_order = []
        for box_row in range(3):
            for box_col in range(3):
                for r in range(box_row*3, (box_row+1)*3):
                    for c in range(box_col*3, (box_col+1)*3):
                        scan_order.append((r, c))
        return scan_order
    
    def _spiral_scan(self) -> List[Tuple[int, int]]:
        """Spiral from outside to inside."""
        scan_order = []
        left, right = 0, 8
        top, bottom = 0, 8
        
        while left <= right and top <= bottom:
            # Top row
            for c in range(left, right + 1):
                scan_order.append((top, c))
            top += 1
            
            # Right column
            for r in range(top, bottom + 1):
                scan_order.append((r, right))
            right -= 1
            
            if top <= bottom:
                # Bottom row
                for c in range(right, left - 1, -1):
                    scan_order.append((bottom, c))
                bottom -= 1
            
            if left <= right:
                # Left column
                for r in range(bottom, top - 1, -1):
                    scan_order.append((r, left))
                left += 1
        
        return scan_order
    
    def _random_walk_scan(self) -> List[Tuple[int, int]]:
        """Natural-looking random walk across the grid."""
        scan_order = []
        visited = set()
        current = (4, 4)  # Start at center
        
        while len(visited) < 81:
            scan_order.append(current)
            visited.add(current)
            
            # Get possible moves (including diagonals)
            moves = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r, c = current[0] + dr, current[1] + dc
                    if 0 <= r < 9 and 0 <= c < 9:
                        moves.append((r, c))
            
            # Prefer unvisited cells with high attention
            unvisited_moves = [m for m in moves if m not in visited]
            if unvisited_moves:
                weights = [self.attention_map[r, c] for r, c in unvisited_moves]
                current = random.choices(unvisited_moves, weights=weights)[0]
            else:
                # If stuck, jump to nearest unvisited cell
                unvisited = set((r, c) for r in range(9) for c in range(9)) - visited
                current = min(unvisited, key=lambda x: ((x[0]-current[0])**2 + 
                                                       (x[1]-current[1])**2))
        
        return scan_order

class PatternLearner:
    """Learns and adapts pattern recognition over time."""
    
    def __init__(self):
        self.pattern_memory = defaultdict(float)  # Pattern frequency
        self.pattern_success = defaultdict(list)  # Pattern success history
        self.difficulty_history = []  # Track difficulty progression
    
    def update_pattern_success(self, pattern: VisualGroup, success: bool):
        """Update success rate for a pattern type."""
        key = (pattern.pattern_type, len(pattern.cells), len(pattern.candidates))
        self.pattern_success[key].append(success)
        
        # Maintain recent history only
        if len(self.pattern_success[key]) > 100:
            self.pattern_success[key].pop(0)
    
    def get_pattern_success_rate(self, pattern: VisualGroup) -> float:
        """Get success rate for a pattern type."""
        key = (pattern.pattern_type, len(pattern.cells), len(pattern.candidates))
        history = self.pattern_success[key]
        if not history:
            return 0.5  # Default for new patterns
        return sum(1 for x in history if x) / len(history)
    
    def adapt_difficulty(self, success_rate: float) -> float:
        """Adapt difficulty based on recent success rate."""
        self.difficulty_history.append(success_rate)
        if len(self.difficulty_history) > 10:
            self.difficulty_history.pop(0)
        
# Calculate trend
        avg_success = sum(self.difficulty_history) / len(self.difficulty_history)
        
        # Adjust difficulty based on success rate
        if avg_success > 0.8:  # Consistently successful
            return min(1.0, self.current_difficulty + 0.1)  # Increase difficulty
        elif avg_success < 0.5:  # Struggling
            return max(0.0, self.current_difficulty - 0.1)  # Decrease difficulty
        else:
            return self.current_difficulty  # Maintain current level
    
    def suggest_next_pattern(self, available_patterns: List[VisualGroup]) -> VisualGroup:
        """Suggest which pattern to focus on next based on learning history."""
        if not available_patterns:
            return None
            
        # Score each pattern based on past success and frequency
        pattern_scores = {}
        for pattern in available_patterns:
            key = (pattern.pattern_type, len(pattern.cells), len(pattern.candidates))
            
            # Calculate success score
            success_rate = self.get_pattern_success_rate(pattern)
            
            # Calculate familiarity score
            familiarity = min(1.0, self.pattern_memory[key] / 50.0)
            
            # Balance between success and novelty
            # Prefer patterns with good success rates but also explore new ones
            exploration_factor = random.random() * 0.2  # 20% random exploration
            score = (success_rate * 0.6 +  # Weight success heavily
                    familiarity * 0.2 +    # Some weight to familiarity
                    pattern.strength * 0.2 + # Consider visual obviousness
                    exploration_factor)     # Add randomness for exploration
            
            pattern_scores[pattern] = score
        
        # Return pattern with highest score
        return max(available_patterns, key=lambda p: pattern_scores[p])
    
    def update_learning_progress(self, patterns: List[VisualGroup], success: bool):
        """Update learning progress for a solving attempt."""
        for pattern in patterns:
            key = (pattern.pattern_type, len(pattern.cells), len(pattern.candidates))
            # Increment pattern frequency
            self.pattern_memory[key] += 1
            # Update success rate
            self.update_pattern_success(pattern, success)
    
    def generate_progress_report(self) -> Dict:
        """Generate a report of learning progress."""
        return {
            'pattern_frequencies': dict(self.pattern_memory),
            'success_rates': {k: sum(v)/len(v) if v else 0 
                            for k, v in self.pattern_success.items()},
            'difficulty_trend': self.difficulty_history[-10:] if self.difficulty_history else []
        }

class AdaptiveLearningSystem:
    """
    Combines visual scanning and pattern learning into an adaptive system.
    """
    
    def __init__(self):
        self.scanner = VisualScanner()
        self.learner = PatternLearner()
        self.current_difficulty = 0.5  # Start at medium difficulty
    
    def analyze_grid(self, grid: np.ndarray, candidates: List[List[Set[int]]]) -> Dict:
        """
        Perform a complete analysis of the grid using learned patterns.
        """
        # Perform visual scan
        scan_result = self.scanner.scan_grid(grid, candidates)
        
        # Get pattern suggestions from learner
        suggested_pattern = self.learner.suggest_next_pattern(scan_result.patterns)
        
        # Adapt difficulty
        success_rate = sum(1 for p in scan_result.patterns 
                         if self.learner.get_pattern_success_rate(p) > 0.7)
        success_rate /= max(1, len(scan_result.patterns))
        self.current_difficulty = self.learner.adapt_difficulty(success_rate)
        
        return {
            'scan_path': scan_result.scan_order,
            'fixation_points': scan_result.fixation_points,
            'detected_patterns': scan_result.patterns,
            'focus_areas': scan_result.focus_areas,
            'suggested_pattern': suggested_pattern,
            'current_difficulty': self.current_difficulty
        }
    
    def update_learning(self, patterns: List[VisualGroup], success: bool):
        """Update learning based on solving attempt."""
        self.learner.update_learning_progress(patterns, success)
    
    def get_learning_state(self) -> Dict:
        """Get current state of learning system."""
        return {
            'attention_map': self.scanner.attention_map.copy(),
            'pattern_memory': dict(self.learner.pattern_memory),
            'current_difficulty': self.current_difficulty,
            'learning_progress': self.learner.generate_progress_report()
        }