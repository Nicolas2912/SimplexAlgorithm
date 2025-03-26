# utils.py
import numpy as np
from fractions import Fraction
import streamlit as st  # <--- ADD THIS LINE


class SolutionStorage:
    """Simple class to store solution details."""
    def __init__(self, decimal_solution, decimal_optimal):
        self.decimal_solution = decimal_solution
        self.decimal_optimal = decimal_optimal

def convert_to_fraction(value, solver=None, fraction_digits=3, force_float=False):
    """
    Convert a decimal value to a fraction string or formatted float.

    Args:
        value: The numerical value to convert.
        solver: The PrimalSimplex solver instance (optional, needed for _limit_fraction).
        fraction_digits: Max digits for numerator/denominator or float precision.
        force_float: If True, always return formatted float.

    Returns:
        Formatted string representation.
    """
    try:
        float_value = float(value)
        if force_float or solver is None:
            return f"{float_value:.{fraction_digits}f}"

        # Use solver's limiting function if available
        frac = solver._limit_fraction(Fraction(float_value))
        num, den = frac.numerator, frac.denominator

        # Limit the size based on fraction_digits
        max_value = 10 ** fraction_digits
        if abs(num) > max_value or den > max_value:
            return f"{float_value:.{fraction_digits}f}"
        return str(frac)

    except (ValueError, TypeError):
        return str(value) # Return original if conversion fails


def create_example_3d():
    """Create a simple example 3D LP problem (Minimize)"""
    c = np.array([-2, -3, -4])
    A = np.array([
        [1, 1, 1],
        [2, 1, 0],
        [0, 1, 3],
    ])
    b = np.array([6, 4, 7])
    return c, A, b

def create_example_2d():
    """Create a simple example 2D LP problem (Minimize)"""
    # Minimize: z = -3x1 - 5x2
    # Subject to:
    #   x1 <= 4
    #   2x2 <= 12  (x2 <= 6)
    #   3x1 + 2x2 <= 18
    #   x1, x2 >= 0
    # Optimal: x1=2, x2=6, z = -36
    c = np.array([-3, -5])
    A = np.array([
        [1, 0],
        [0, 2],
        [3, 2]
    ])
    b = np.array([4, 12, 18])
    return c, A, b


def validate_inputs(c, A, b, m, n):
    """Validate input dimensions and values more thoroughly."""
    try:
        # Type checks
        if not isinstance(c, np.ndarray) or c.ndim != 1: return False, "Objective coefficients (c) must be a 1D array."
        if not isinstance(A, np.ndarray) or A.ndim != 2: return False, "Constraint matrix (A) must be a 2D array."
        if not isinstance(b, np.ndarray) or b.ndim != 1: return False, "RHS values (b) must be a 1D array."

        # Dimension consistency
        actual_m, actual_n = A.shape
        if actual_n != n: return False, f"Number of variables (n={n}) doesn't match A's columns ({actual_n})."
        if actual_m != m: return False, f"Number of constraints (m={m}) doesn't match A's rows ({actual_m})."
        if len(c) != n: return False, f"Length of objective coefficients c ({len(c)}) doesn't match n ({n})."
        if len(b) != m: return False, f"Length of RHS values b ({len(b)}) doesn't match m ({m})."

        # Value checks (optional, but good)
        if not np.all(np.isfinite(c)): return False, "Objective coefficients (c) contain non-finite values (NaN or Inf)."
        if not np.all(np.isfinite(A)): return False, "Constraint matrix (A) contains non-finite values (NaN or Inf)."
        if not np.all(np.isfinite(b)): return False, "RHS values (b) contain non-finite values (NaN or Inf)."

        return True, "Inputs are valid."

    except Exception as e:
        return False, f"Input validation error: {e}"

def initialize_session_state():
    """Initialize all required session state variables if they don't exist."""
    # Load default 3D example values initially
    c_def, A_def, b_def = create_example_3d()
    m_def, n_def = A_def.shape

    defaults = {
        'solution_storage': None,
        'tableaus': [],
        'has_solved': False,
        'solver': None,
        'original_params': None,
        'sensitivity_resample': None,
        'auto_solve': False,
        'active_tab_index': 0,
        'coming_from_whatif': False,
        # REMOVED 'use_example': True,
        # Default to 3D example dimensions and values
        'm': m_def,
        'n': n_def,
        'c_input': ",".join(map(str, c_def.astype(float))),
        'A_input': "\n".join(",".join(map(str, row.astype(float))) for row in A_def),
        'b_input': ",".join(map(str, b_def.astype(float))),
        # Other defaults
        'use_fractions': True,
        'fraction_digits': 3,
        'eq_constraints': False,
        'show_progress': False,
        'problem_params': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value