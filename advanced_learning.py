# advanced_learning.py

import json
import pickle
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path
import sqlite3
from collections import defaultdict

@dataclass
class LearnedPattern:
    """Representation of a learned pattern with metadata."""
    pattern_type: str
    cells: List[Tuple[int, int]]
    candidates: Set[int]
    success_rate: float
    discovery_date: datetime
    last_used: datetime
    usage_count: int
    avg_solve_time: float
    difficulty_rating: float
    prerequisites: List[str]  # Patterns that usually come before this one
    variations: List[Dict]    # Similar patterns with slight differences

@dataclass
class StrategyEffectiveness:
    """Detailed tracking of strategy effectiveness."""
    strategy_name: str
    success_count: int
    failure_count: int
    avg_time_taken: float
    difficulty_level: float
    prerequisite_patterns: List[str]
    typical_grid_states: List[Dict]  # Representative grid states where strategy works
    last_used: datetime
    confidence_score: float

class PatternDatabase:
    """Manages persistent storage of learned patterns and strategy data."""
    
    def __init__(self, db_path: str = "learned_patterns.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY,
                    pattern_type TEXT,
                    pattern_data BLOB,
                    success_rate REAL,
                    discovery_date TEXT,
                    last_used TEXT,
                    usage_count INTEGER,
                    avg_solve_time REAL,
                    difficulty_rating REAL
                )
            """)
            
            # Strategy effectiveness table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_effectiveness (
                    id INTEGER PRIMARY KEY,
                    strategy_name TEXT,
                    effectiveness_data BLOB,
                    last_updated TEXT
                )
            """)
            
            # Pattern relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_relationships (
                    pattern_id INTEGER,
                    prerequisite_id INTEGER,
                    relationship_type TEXT,
                    strength REAL,
                    FOREIGN KEY (pattern_id) REFERENCES patterns(id),
                    FOREIGN KEY (prerequisite_id) REFERENCES patterns(id)
                )
            """)
            
            conn.commit()
    
    def save_pattern(self, pattern: LearnedPattern):
        """Save or update a learned pattern."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Serialize pattern data
            pattern_data = pickle.dumps(asdict(pattern))
            
            # Update if exists, insert if new
            cursor.execute("""
                INSERT OR REPLACE INTO patterns (
                    pattern_type, pattern_data, success_rate, discovery_date,
                    last_used, usage_count, avg_solve_time, difficulty_rating
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_type,
                pattern_data,
                pattern.success_rate,
                pattern.discovery_date.isoformat(),
                pattern.last_used.isoformat(),
                pattern.usage_count,
                pattern.avg_solve_time,
                pattern.difficulty_rating
            ))
            
            conn.commit()
    
    def load_patterns(self) -> List[LearnedPattern]:
        """Load all learned patterns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pattern_data FROM patterns")
            return [
                LearnedPattern(**pickle.loads(row[0]))
                for row in cursor.fetchall()
            ]
    
    def update_strategy_effectiveness(self, effectiveness: StrategyEffectiveness):
        """Update strategy effectiveness data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            effectiveness_data = pickle.dumps(asdict(effectiveness))
            
            cursor.execute("""
                INSERT OR REPLACE INTO strategy_effectiveness (
                    strategy_name, effectiveness_data, last_updated
                ) VALUES (?, ?, ?)
            """, (
                effectiveness.strategy_name,
                effectiveness_data,
                datetime.now().isoformat()
            ))
            
            conn.commit()

class AdvancedPatternDiscovery:
    """Advanced pattern discovery and analysis system."""
    
    def __init__(self):
        self.known_patterns = set()
        self.pattern_relationships = defaultdict(list)
    
    def discover_new_patterns(self, grid: np.ndarray, 
                            candidates: List[List[Set[int]]]) -> List[LearnedPattern]:
        """
        Discover new patterns using advanced techniques.
        """
        new_patterns = []
        
        # Multiple discovery methods
        new_patterns.extend(self._discover_structural_patterns(grid, candidates))
        new_patterns.extend(self._discover_candidate_patterns(grid, candidates))
        new_patterns.extend(self._discover_composite_patterns(grid, candidates))
        
        # Filter out known patterns
        truly_new = [p for p in new_patterns if self._is_novel_pattern(p)]
        
        # Update known patterns
        self.known_patterns.update(
            (p.pattern_type, tuple(p.cells), tuple(sorted(p.candidates))) 
            for p in truly_new
        )
        
        return truly_new
    
    def _discover_structural_patterns(self, grid: np.ndarray, 
                                    candidates: List[List[Set[int]]]) -> List[LearnedPattern]:
        """Discover patterns based on structural relationships."""
        patterns = []
        
        # Look for symmetric patterns
        patterns.extend(self._find_symmetric_patterns(grid))
        
        # Look for chains and cycles
        patterns.extend(self._find_chain_patterns(grid, candidates))
        
        # Look for intersection patterns
        patterns.extend(self._find_intersection_patterns(grid, candidates))
        
        return patterns
    
    def _discover_candidate_patterns(self, grid: np.ndarray,
                                   candidates: List[List[Set[int]]]) -> List[LearnedPattern]:
        """Discover patterns in candidate distributions."""
        patterns = []
        
        # Analyze candidate frequency distributions
        freq_patterns = self._analyze_candidate_frequencies(candidates)
        patterns.extend(freq_patterns)
        
        # Look for candidate chains
        chain_patterns = self._find_candidate_chains(candidates)
        patterns.extend(chain_patterns)
        
        return patterns
    
    def _discover_composite_patterns(self, grid: np.ndarray,
                                   candidates: List[List[Set[int]]]) -> List[LearnedPattern]:
        """Discover complex patterns composed of simpler ones."""
        patterns = []
        
        # Find combinations of known patterns
        for p1 in self.known_patterns:
            for p2 in self.known_patterns:
                if p1 != p2:
                    combined = self._try_combine_patterns(p1, p2, grid, candidates)
                    if combined:
                        patterns.append(combined)
        
        return patterns

class AdvancedLearningSystem:
    """Advanced learning system with persistence and sophisticated pattern discovery."""
    
    def __init__(self, db_path: str = "sudoku_learning.db"):
        self.pattern_db = PatternDatabase(db_path)
        self.pattern_discovery = AdvancedPatternDiscovery()
        self.strategy_effectiveness = {}
        self.load_learned_data()
    
    def load_learned_data(self):
        """Load previously learned patterns and strategy data."""
        self.learned_patterns = self.pattern_db.load_patterns()
        # Initialize strategy effectiveness tracking
        for pattern in self.learned_patterns:
            self._update_strategy_relationships(pattern)
    
    def update_learning(self, grid: np.ndarray, candidates: List[List[Set[int]]],
                       strategy_used: str, success: bool, time_taken: float):
        """Update learning with new solving attempt data."""
        # Discover any new patterns
        new_patterns = self.pattern_discovery.discover_new_patterns(grid, candidates)
        
        # Update pattern database
        for pattern in new_patterns:
            pattern.discovery_date = datetime.now()
            pattern.last_used = datetime.now()
            pattern.usage_count = 1
            pattern.success_rate = 1.0 if success else 0.0
            pattern.avg_solve_time = time_taken
            self.pattern_db.save_pattern(pattern)
        
        # Update strategy effectiveness
        self._update_strategy_effectiveness(strategy_used, success, time_taken, grid)
        
        # Save updates
        self._persist_learning_updates()
    
    def _update_strategy_effectiveness(self, strategy: str, success: bool, 
                                     time_taken: float, grid: np.ndarray):
        """Update detailed strategy effectiveness data."""
        if strategy not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy] = StrategyEffectiveness(
                strategy_name=strategy,
                success_count=0,
                failure_count=0,
                avg_time_taken=0.0,
                difficulty_level=0.5,
                prerequisite_patterns=[],
                typical_grid_states=[],
                last_used=datetime.now(),
                confidence_score=0.5
            )
        
        effectiveness = self.strategy_effectiveness[strategy]
        
        # Update counts and averages
        if success:
            effectiveness.success_count += 1
        else:
            effectiveness.failure_count += 1
        
        total_attempts = effectiveness.success_count + effectiveness.failure_count
        effectiveness.avg_time_taken = (
            (effectiveness.avg_time_taken * (total_attempts - 1) + time_taken) 
            / total_attempts
        )
        
        # Update typical grid states
        grid_state = grid.copy().tolist()
        effectiveness.typical_grid_states.append({
            'grid': grid_state,
            'success': success,
            'time_taken': time_taken
        })
        
        # Keep only most recent states
        effectiveness.typical_grid_states = effectiveness.typical_grid_states[-10:]
        
        # Update confidence score
        effectiveness.confidence_score = (
            effectiveness.success_count / total_attempts * 
            min(1.0, total_attempts / 10.0)  # Scale by experience
        )
        
        # Save updates
        self.pattern_db.update_strategy_effectiveness(effectiveness)
    
    def get_learning_state(self) -> Dict:
        """Get current state of the learning system."""
        return {
            'total_patterns_learned': len(self.learned_patterns),
            'strategy_effectiveness': {
                name: {
                    'success_rate': (
                        strat.success_count / 
                        (strat.success_count + strat.failure_count)
                    ),
                    'confidence': strat.confidence_score,
                    'avg_time': strat.avg_time_taken
                }
                for name, strat in self.strategy_effectiveness.items()
            },
            'recent_discoveries': [
                {
                    'type': pattern.pattern_type,
                    'success_rate': pattern.success_rate,
                    'discovered': pattern.discovery_date.isoformat()
                }
                for pattern in sorted(
                    self.learned_patterns,
                    key=lambda p: p.discovery_date,
                    reverse=True
                )[:5]
            ]
        }
    
    def _persist_learning_updates(self):
        """Save all learning updates to persistent storage."""
        for pattern in self.learned_patterns:
            self.pattern_db.save_pattern(pattern)
        for effectiveness in self.strategy_effectiveness.values():
            self.pattern_db.update_strategy_effectiveness(effectiveness)