# Sudoku Solver with Step-Based (Hint-Based) Training

This repository demonstrates a Sudoku solver that learns to fill one cell at a time,
in a more "human-like" step-by-step manner. 

## How it Works

- **Step Data**: Each Sudoku puzzle is broken into multiple "partial states," each with 
  exactly one cell to fill next.
- **Model Architecture**: The model outputs three separate logits: 
  **(row, col, digit)** for the next cell to fill.
- **GUI**: There's a PyQt6 GUI to generate a puzzle and solve it step by step.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
