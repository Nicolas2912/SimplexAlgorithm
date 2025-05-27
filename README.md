# Simplex Algorithm & LP Solver

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Nicolas2912/SimplexAlgorithm)

![Linear Programming Solver Demo](app_record.gif)

A comprehensive optimization toolkit, including:
- Primal Simplex algorithm (with Bland's anti-cycling rule)
- Karmarkar's Interior Point method
- Branch & Bound solver for 0/1 Knapsack problems
- SciPy and Gurobi integrations for LP/MIP solving and verification
- Google Gemini LLM-assisted LP formulation and code generation
All accessible through an interactive Streamlit web interface.

## Features

- Multiple solver implementations:
  - Primal Simplex algorithm (with Bland's anti-cycling rule)
  - SciPy and Gurobi-based LP/MIP solvers for comparison and verification
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
SimplexAlgorithm/
├── README.md
├── Simplex_Frontend.gif
├── app_record.gif
├── branch_and_bound.py          # Branch & Bound solver for knapsack problems
├── debug/                       # Debugging scripts and utilities
│   ├── debug_compare.py
│   ├── debug_simplex.py
│   └── debug_tableau.py
├── frontend_simplex.py          # Streamlit web interface (legacy)
├── simplex_solver.py            # Main Streamlit application entry point
├── gurobi_solver.py             # Gurobi-based MIP solver example
├── inner_point.py               # Karmarkar's Interior Point algorithm
├── pages/                       # Additional Streamlit pages
│   └── llm_lp_solver.py         # LLM-assisted LP formulation and code generation
├── plotting.py                  # Visualization functions for LP problems
├── sensitivity_ui.py            # Streamlit UI for sensitivity analysis
├── simplex.py                   # Core Primal Simplex implementation
├── ui_components.py             # Reusable UI components for Streamlit interface
├── utils.py                     # Utility functions and helpers
├── test_blands_rule.py          # Demonstration of Bland's anti-cycling rule
├── test_simplex.py              # Test suite for validating solver implementations
├── requirements.txt             # Project dependencies
└── .devcontainer/               # Development container configuration
    └── devcontainer.json
```

## Installation

1. Clone the repository:
```powershell
git clone https://github.com/Nicolas2912/SimplexAlgorithm.git
cd SimplexAlgorithm
```

2. Create and activate a conda environment:
```powershell
conda create -n lp-solver python=3.12
conda activate lp-solver
```

3. Install dependencies:
```powershell
pip install -r requirements.txt

4. (Optional) Configure the LLM LP Solver:
```bash
echo GEMINI_API_KEY=your_api_key_here > .env
```
```

## Usage

The easiest way is to visit my apps website on streamlit:

`https://simplex-algorithm.streamlit.app/`


### Running the Web Interface locally

1. Start the Streamlit application:
```powershell
python -m streamlit run simplex_solver.py
```

or for other shells:
```bash
python -m streamlit run simplex_solver.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

#### LLM LP Solver Page

To use the Google Gemini LLM-based LP solver interface, ensure you have a valid `GEMINI_API_KEY` configured (see Installation), then run:

```bash
python -m streamlit run pages/llm_lp_solver.py
```

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

The project includes a comprehensive test suite that validates solver implementations:

```bash
python -m unittest discover -v
```

Tests include:
- Simple 2D and 3D LP problems
- Equality and inequality constraints
- Unbounded and infeasible problems
- Problems with numerical stability challenges
- Comparison with SciPy's LP solver implementation
- Bland's anti-cycling rule demonstration script

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
