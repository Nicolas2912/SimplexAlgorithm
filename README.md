# Linear Programming Solver

![Linear Programming Solver Demo](app_record.gif)

A comprehensive implementation of Linear Programming algorithms, featuring both the Primal Simplex method and Interior Point method for solving Linear Programming problems. This project includes a robust backend solver implementation and a user-friendly web interface built with Streamlit.

## Features

- Multiple LP solver implementations:
  - Two-Phase Primal Simplex algorithm
  - Karmarkar's Interior Point method
- Comprehensive support for LP problem types:
  - Equality constraints
  - Inequality constraints
  - Mixed constraint types
- Advanced analytical capabilities:
  - Sensitivity analysis for objective coefficients and right-hand side values
  - Shadow price calculations
  - What-if scenario testing
- Interactive visualization features:
  - 2D and 3D problem visualization
  - Step-by-step solution with pivot information
  - Solution path visualization
  - Dynamic parameter adjustments with instant feedback
- Robust implementation:
  - Comprehensive test suite with comparison to SciPy's LP solver
  - Numerical stability enhancements
  - Support for both decimal and fraction output formats
- User-friendly interface:
  - Example problems included for learning and testing
  - LaTeX formatting for mathematical expressions
  - Intuitive input methods for problem specification


## Project Structure

```
linear-programming-solver/
├── simplex.py           # Core implementation of the Primal Simplex algorithm
├── inner_point.py       # Implementation of Karmarkar's Interior Point algorithm
├── sensitivity_ui.py    # UI components for sensitivity analysis
├── ui_components.py     # Reusable UI components for Streamlit interface
├── plotting.py          # Visualization functions for LP problems
├── utils.py             # Utility functions and helpers
├── app.py               # Main Streamlit application entry point
├── frontend_simplex.py  # Streamlit web interface
├── test_simplex.py      # Test suite for validating solver implementations
└── requirements.txt     # Project dependencies
```

## Installation

1. Clone the repository:
```powershell
git clone https://github.com/yourusername/linear-programming-solver.git
cd linear-programming-solver
```

2. Create and activate a conda environment:
```powershell
conda create -n lp-solver python=3.12
conda activate lp-solver
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage

The easiest way is to visit my apps website on streamlit:

`https://simplex-algorithm.streamlit.app/`


### Running the Web Interface locally

1. Start the Streamlit application (powershell):
```powershell
python -m streamlit run app.py
```

or for other shells:
```bash
python run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

### Using the Command Line Interface

You can also use the solvers directly through Python:

#### Primal Simplex Solver

```python
import numpy as np
from simplex import PrimalSimplex

# Define your problem
c = np.array([2, 3])  # Objective function coefficients
A = np.array([[1, 2], [2, 1], [1, 1]])  # Constraint coefficients
b = np.array([10, 8, 5])  # Right-hand side values

# Create solver instance
# For inequality constraints (≤)
solver = PrimalSimplex(c, A, b, use_fractions=True, eq_constraints=False)

# Or for equality constraints (=)
solver = PrimalSimplex(c, A, b, use_fractions=True, eq_constraints=True)

# Solve the problem
solution, optimal_value = solver.solve()

print("Optimal solution:", solution)
print("Optimal value:", optimal_value)
```

#### Interior Point Solver

```python
import numpy as np
from inner_point import KarmarkarSolver

# Define your problem in Karmarkar form
# A*x = 0, sum(x) = 1, x > 0
A = np.array([[2, -1, -1]])
c = np.array([[-1], [-1], [0]])

# Starting point within the feasible region (must satisfy constraints)
x0 = np.array([[1/3], [1/3], [1/3]])

# Create and solve
solver = KarmarkarSolver(A, c, sense='minimize', alpha=0.5)
solution, iterations, optimal_value, stop_reason, path = solver.solve(x0)

print("Optimal solution:", solution.flatten())
print("Optimal value:", optimal_value)
print("Path length:", len(path))
```

#### Sensitivity Analysis

```python
from simplex import PrimalSimplex, SensitivityAnalysis

# After solving with PrimalSimplex
solver = PrimalSimplex(c, A, b)
solution, optimal_value = solver.solve()

# Perform sensitivity analysis
analysis = SensitivityAnalysis(solver)

# Get allowable ranges for objective coefficients
obj_ranges = analysis.objective_sensitivity_analysis()

# Get allowable ranges for RHS values
rhs_ranges = analysis.rhs_sensitivity_analysis()

# Get shadow prices
shadow_prices = analysis.shadow_prices()

print("Objective coefficient ranges:", obj_ranges)
print("RHS ranges:", rhs_ranges)
print("Shadow prices:", shadow_prices)
```

## Testing

The project includes a comprehensive test suite that validates the correctness of the simplex implementation:

```powershell
python -m unittest test_simplex.py
```

Tests include:
- Simple 2D and 3D LP problems
- Equality and inequality constraints
- Unbounded and infeasible problems
- Problems with numerical stability challenges
- Comparison with SciPy's LP solver implementation

## Problem Format

The solver handles two types of constraint systems based on the `eq_constraints` parameter:

### Standard Form (eq_constraints=False)
All constraints are treated as less than or equal (≤) constraints. The problem follows the standard form:

$$
\begin{align*}
\text{minimize} \quad & z = \mathbf{c}^\top \mathbf{x} \\
\text{subject to} \quad & \mathbf{A}\mathbf{x} \leq \mathbf{b} \\
& \mathbf{x} \geq \mathbf{0}
\end{align*}
$$

Or more explicitly:

$$
\begin{align*}
\text{minimize} \quad & z = \sum_{j=1}^n c_j x_j \\
\text{subject to} \quad & \sum_{j=1}^n a_{ij}x_j \leq b_i, \quad i = 1,\ldots,m \\
& x_j \geq 0, \quad j = 1,\ldots,n
\end{align*}
$$

### Equality Form (eq_constraints=True)
All constraints are treated as equality (=) constraints, using the two-phase simplex method:

$$
\begin{align*}
\text{minimize} \quad & z = \mathbf{c}^\top \mathbf{x} \\
\text{subject to} \quad & \mathbf{A}\mathbf{x} = \mathbf{b} \\
& \mathbf{x} \geq \mathbf{0}
\end{align*}
$$

Or more explicitly:

$$
\begin{align*}
\text{minimize} \quad & z = \sum_{j=1}^n c_j x_j \\
\text{subject to} \quad & \sum_{j=1}^n a_{ij}x_j = b_i, \quad i = 1,\ldots,m \\
& x_j \geq 0, \quad j = 1,\ldots,n
\end{align*}
$$

### Karmarkar Form (Interior Point)
For the Interior Point method, problems must be in Karmarkar's standard form:

$$
\begin{align*}
\text{minimize} \quad & z = \mathbf{c}^\top \mathbf{x} \\
\text{subject to} \quad & \mathbf{A}\mathbf{x} = \mathbf{0} \\
& \sum_{j=1}^n x_j = 1 \\
& \mathbf{x} > \mathbf{0}
\end{align*}
$$

Where:
- $\mathbf{x} \in \mathbb{R}^n$ is the vector of decision variables
- $\mathbf{c} \in \mathbb{R}^n$ is the cost vector
- $\mathbf{A} \in \mathbb{R}^{m \times n}$ is the constraint coefficient matrix
- $\mathbf{b} \in \mathbb{R}^m$ is the right-hand side vector (for simplex methods)

Note: When using `eq_constraints=True`, the solver automatically implements the two-phase simplex method to handle the equality constraints, introducing artificial variables as needed.

## Authors

Nicolas Schneider

## Acknowledgments

- Implementation based on the Two-Phase Simplex Method and Karmarkar's Interior Point Algorithm
- Web interface built using Streamlit
- Mathematical formulations from Linear Programming literature
