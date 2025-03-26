# ui_components.py
import streamlit as st
import pandas as pd
from fractions import Fraction
import numpy as np
from utils import convert_to_fraction # Import from utils

def format_lp_problem(c, A, b, n, m, eq_constraints=False):
    """
    Format the LP problem in LaTeX.

    Parameters are assumed to be valid numpy arrays/ints by the time this is called.
    """
    latex = r"\begin{align*}"

    # Objective Function
    obj_terms = []
    for i in range(n):
        coeff = c[i]
        term = ""
        # Format coefficient
        if np.isclose(coeff, 0):
            continue # Skip zero terms
        elif np.isclose(abs(coeff), 1):
            sign = "+" if coeff > 0 else "-"
            term = f"{sign}x_{{{i + 1}}}"
            if i == 0 and coeff > 0: # Remove leading '+' for the first term
                 term = f"x_{{{i + 1}}}"
        else:
            # Use format specifier for potentially cleaner output
            formatted_coeff = f"{coeff:+.2g}" if coeff > 0 else f"{coeff:.2g}"
            term = f"{formatted_coeff}x_{{{i + 1}}}"
            if i == 0 and coeff > 0: # Remove leading '+' for the first term
                 term = f"{coeff:.2g}x_{{{i + 1}}}"

        obj_terms.append(term)

    if not obj_terms: obj_terms.append("0") # Handle case of zero objective

    latex += f"\\min \\quad & z = {' '.join(obj_terms)} \\\\[1em]"
    latex += r"\text{s.t.} \quad & "

    # Constraints
    constraint_symbol = "=" if eq_constraints else r"\leq"
    for i in range(m):
        constraint_terms = []
        for j in range(n):
            coeff = A[i, j]
            term = ""
            # Format coefficient
            if np.isclose(coeff, 0):
                continue
            elif np.isclose(abs(coeff), 1):
                sign = "+" if coeff > 0 else "-"
                term = f"{sign}x_{{{j + 1}}}"
                if not constraint_terms and coeff > 0: # First term in constraint
                     term = f"x_{{{j + 1}}}"
            else:
                 formatted_coeff = f"{coeff:+.2g}" if coeff > 0 else f"{coeff:.2g}"
                 term = f"{formatted_coeff}x_{{{j + 1}}}"
                 if not constraint_terms and coeff > 0: # First term in constraint
                     term = f"{coeff:.2g}x_{{{j + 1}}}"

            constraint_terms.append(term)

        constraint_str = ' '.join(constraint_terms) if constraint_terms else "0"
        latex += f"\qquad {constraint_str} {constraint_symbol} {b[i]:.2g}" # Format b value

        if i < m - 1: latex += r" \\ & " # Add alignment for next row

    # Non-negativity
    var_indices = ", ".join([str(j+1) for j in range(n)])
    latex += r" \\ & \qquad x_j \geq 0 \quad \forall j \in \{" + var_indices + r"\}"
    latex += r"\end{align*}"
    return latex


def display_iteration(iteration, tableau, solver, use_fractions, fraction_digits=3, pivot_info=None):
    """Display the tableau for a given iteration with formatting."""
    st.write(f"**Iteration {iteration}**")

    if pivot_info:
        entering_var_idx, leaving_row_idx = pivot_info
        # Determine variable name (decision or slack)
        entering_var_name = f"x_{entering_var_idx + 1}" if entering_var_idx < solver.n else f"s_{entering_var_idx - solver.n + 1}"
        st.write(f"Pivot: Entering Variable = ${entering_var_name}$, Leaving Row = $R_{leaving_row_idx}$ (basis variable leaving)")

    headers = ["Basis"] + [f"x{i + 1}" for i in range(solver.n)] + \
              [f"s{i + 1}" for i in range(solver.m)] + ["RHS"]

    # Identify basis variables for row labels
    basis_vars = ["z"] # Row 0 is always z
    for r in range(1, solver.m + 1):
        basis_var_name = "N/A"
        for c in range(solver.n + solver.m): # Check all var columns
            col = tableau[1:, c] # Look only in constraint rows
            # Check if column 'c' is basic in row 'r' (relative to constraint rows)
            if np.isclose(tableau[r, c], 1.0) and np.all(np.isclose(np.delete(tableau[:, c], r), 0.0)):
                 basis_var_name = f"x_{c + 1}" if c < solver.n else f"s_{c - solver.n + 1}"
                 break # Found basis var for this row
        basis_vars.append(basis_var_name)

    # Format tableau data
    formatted_data = []
    for i in range(tableau.shape[0]):
        row_data = [basis_vars[i]] # Add basis var name as first element
        for j in range(tableau.shape[1]):
            # Use convert_to_fraction, force float for non-fraction mode
            formatted_val = convert_to_fraction(
                tableau[i, j],
                solver=solver if use_fractions else None,
                fraction_digits=fraction_digits,
                force_float=not use_fractions
            )
            row_data.append(formatted_val)
        formatted_data.append(row_data)

    df = pd.DataFrame(formatted_data, columns=headers)

    # Use st.dataframe for better table rendering
    st.dataframe(df.style.set_properties(**{
        'text-align': 'right',
        # 'font-family': 'monospace', # Often looks good
    }), hide_index=True, use_container_width=True)


def display_solution(solver, solution_storage, use_fractions, fraction_digits):
    """Display the final optimal solution in LaTeX."""
    st.header("Optimal Solution")

    if solution_storage is None or solver is None:
        st.warning("Solution data not available.")
        return

    solution_latex = "\\begin{align*}\n"

    # Format solution vector x*
    x_star_vals = []
    for i in range(solver.n):
        val = solution_storage.decimal_solution[i]
        formatted_val = convert_to_fraction(
            val,
            solver=solver if use_fractions else None,
            fraction_digits=fraction_digits,
            force_float=not use_fractions
        )
        x_star_vals.append(f"x_{{{i+1}}}^* &= {formatted_val}")

    # Format optimal value z*
    z_optimal_val = solution_storage.decimal_optimal
    z_optimal_formatted = convert_to_fraction(
        z_optimal_val,
        solver=solver if use_fractions else None,
        fraction_digits=fraction_digits + 2, # Allow more precision for z*
        force_float=not use_fractions
    )

    solution_latex += "\\mathbf{x^*} = \\begin{cases}" + " \\\\ ".join(x_star_vals) + "\\end{cases}"
    solution_latex += f" & \\qquad z^* = {z_optimal_formatted}"
    solution_latex += "\\end{align*}"

    st.latex(solution_latex)