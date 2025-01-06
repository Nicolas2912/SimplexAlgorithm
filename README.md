# Linear Programming Solver

A robust implementation of the Simplex algorithm for solving linear programming problems, featuring both primal and dual simplex methods with an interactive web interface.

## Overview

This project provides a comprehensive solution for solving linear programming problems using different variants of the Simplex algorithm. It consists of three main components:

- `simplex.py`: Core implementation of the Primal and Dual Simplex algorithms
- `test_simplex.py`: Extensive test suite ensuring algorithmic correctness
- `frontend_simplex.py`: Streamlit-based web interface for interactive problem-solving

## Features

- **Multiple Solver Methods**
  - Primal Simplex algorithm with two-phase method support
  - Dual Simplex algorithm
  - Integration with SciPy's linear programming solver

- **Robust Problem Handling**
  - Support for equality and inequality constraints
  - Automatic handling of negative right-hand side values
  - Detection of unbounded and infeasible problems
  - Handling of degenerate cases

- **Interactive Web Interface**
  - Real-time problem visualization
  - Step-by-step solution process
  - Support for both decimal and fraction output
  - Dynamic tableau visualization
  - LaTeX rendering of mathematical formulations

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Streamlit
- Pandas
- Tabulate

## Installation

```bash
pip install numpy scipy streamlit pandas tabulate
```

## Usage

### Command Line Interface
```python
from simplex import PrimalSimplex
import numpy as np

# Define your linear programming problem
c = np.array([2, 3])               # Objective function coefficients
A = np.array([[1, 2], [2, 1]])     # Constraint coefficients
b = np.array([10, 8])              # Right-hand side values

# Create and solve
solver = PrimalSimplex(c, A, b)
solution, optimal_value = solver.solve()
```

### Web Interface
```bash
streamlit run frontend_simplex.py
```

## Testing

The project includes comprehensive unit tests covering various scenarios:
```bash
python -m unittest test_simplex.py
```

## Mathematical Background

The implementation solves linear programming problems in the standard form:

```
Minimize    c^T x
Subject to  Ax ≤ b
           x ≥ 0
```

For equality constraints, the solver automatically handles the problem using the two-phase simplex method.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.