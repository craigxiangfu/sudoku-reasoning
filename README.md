**Status:** In Development - Part II

# Sudoku Reasoning with Step-Based (Hint-Based) Training (Part I)

Reasoning Model Group, Modularium Research, December 2024

This repo demonstrates a Sudoku solving system that trys (its best) to mimic human reasoning. It clearns to fill one cell at a time,
in a more "human-like" step-by-step manner. But this is only the bottom line of "human reasoning".

## Why It’s Not Truly Human Reasoning

**Heuristics vs. Cognition**: This system encodes a set of logical heuristics (e.g., “hidden pairs,” “X-Wing,” “swordfish” patterns, etc.) that have been observed in human solvers. Humans typically learn these heuristics over time by trial, error, and memory. However, actual human reasoning also involves intuition, pattern-matching gleaned from real-world experiences, internal mental imagery, and even meta-cognitive processes (knowing when or why you are stuck, how to guess, how to pivot strategies, etc.). Software logic—like what’s in this code—cannot capture all of that richness.

**No True Introspection**: A fundamental feature of human thought is introspection: you not only apply logical rules but also become aware of the application of these rules, your confidence level, your frustration, or your sense of “aha!” when something clicks. The code you provided tracks “success_rate,” “confidence_score,” or “usage_count,” but these are numeric abstractions without subjective experience or true self-awareness.

**Rule Execution vs. Embodied Reasoning**: Humans are influenced by context, visual scanning idiosyncrasies, memory constraints, and emotional states. A large portion of “human reasoning” in puzzle-solving is shaped by these “extraneous” factors. The code’s “visual scanning” and “pattern discovery” modules mimic a scanning or attention mechanism, but they remain a set of algorithms. They do not integrate physical constraints (like eye fatigue), personal puzzle preference, or real-time neural processing.

**Pre-programmed Logic**: All the strategies—like pointing pairs, hidden pairs, box-line reduction—are pre-programmed. While the system can adapt success rates or refine which strategies it tries first, it does not spontaneously discover an entirely new strategy unprompted the way a human might. It merely rearranges priorities within a fixed set of known strategies.

**No Genuine Cognitive Architecture**: Human reasoning occurs within a brain that manages sense perceptions, emotional responses, linguistic reasoning, and more. This code is deterministic or probabilistic logic layered on data structures—something that is useful for tasks like Sudoku but does not replicate the complexity of neural cognition.

## How it Works at the Moment

- **Step Data**: Each Sudoku puzzle is broken into multiple "partial states," each with 
  exactly one cell to fill next.
- **Model Architecture**: The model outputs three separate logits: 
  **(row, col, digit)** for the next cell to fill.
- **GUI**: There's a PyQt6 GUI to generate a puzzle and solve it step by step.