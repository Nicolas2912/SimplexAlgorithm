import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from tabulate import tabulate
from simplex import PrimalSimplex, SensitivityAnalysis  # Assuming class is in simplex.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.patches as patches  # Keep for 2D polygon


class SolutionStorage:
    def __init__(self, decimal_solution, decimal_optimal):
        self.decimal_solution = decimal_solution
        self.decimal_optimal = decimal_optimal


def convert_to_fraction(value, solver, fraction_digits):
    """Convert a decimal value to a fraction string with given digit limit"""
    if solver is None: return f"{float(value):.{fraction_digits}f}"  # Handle solver being None
    frac = solver._limit_fraction(Fraction(float(value)))
    # Get numerator and denominator
    num, den = frac.numerator, frac.denominator
    # Limit the size of both numerator and denominator based on fraction_digits
    max_value = 10 ** fraction_digits
    if abs(num) > max_value or den > max_value:
        return f"{float(value):.{fraction_digits}f}"
    return str(frac)


def create_example_problem():
    """Create a simple example 3D LP problem"""
    # Minimize: -2x1 - 3x2 - 4x3
    # Subject to:
    #   x1 + x2 + x3 <= 6
    #   2*x1 + x2 + 0*x3 <= 4
    #   0*x1 + x2 + 3*x3 <= 7
    #   x1, x2, x3 >= 0
    # Optimal solution should be around x = [0.5, 3, 4/3], z = -15.33
    c = np.array([-2, -3, -4])
    A = np.array([
        [1, 1, 1],
        [2, 1, 0],
        [0, 1, 3],
    ])
    b = np.array([6, 4, 7])
    return c, A, b


def format_lp_problem(c, A, b, n, m, eq_constraints=False):
    """
    Format the LP problem in LaTeX.
    
    Parameters:
    -----------
    c : array-like
        Coefficients of the objective function (length n)
    A : array-like
        Constraint coefficients matrix (m x n)
    b : array-like
        Right-hand side of constraints (length m)
    n : int
        Number of variables
    m : int
        Number of constraints
    eq_constraints : bool, optional
        Whether to use equality constraints (=) instead of inequalities (≤)
    """
    # Ensure dimensions are consistent
    actual_m, actual_n = A.shape
    n = min(n, actual_n)  # Use smaller of provided n or actual columns in A
    m = min(m, actual_m)  # Use smaller of provided m or actual rows in A
    
    latex = r"\begin{align*}"
    obj_terms = []
    for i in range(min(n, len(c))):
         coeff = c[i]
         term = f"{coeff:+}x_{{{i + 1}}}" if coeff >=0 else f"{coeff}x_{{{i + 1}}}"
         # Simplify +1 or -1 coefficients
         if abs(coeff) == 1:
              term = f"+x_{{{i + 1}}}" if coeff == 1 else f"-x_{{{i + 1}}}"
              if i == 0 and coeff == 1: term = f"x_{{{i + 1}}}" # No plus sign for first term if positive
         obj_terms.append(term)

    latex += f"\\min \\quad & z = {' '.join(obj_terms)} \\\\[1em]"
    latex += r"\text{s.t.} \quad & "
    constraint_symbol = "=" if eq_constraints else r"\leq"

    for i in range(m):
        constraint_terms = []
        for j in range(n):
             if i < actual_m and j < actual_n:  # Check bounds before accessing A
                 coeff = A[i, j]
                 if abs(coeff) > 1e-10: # Only include non-zero terms
                      term = f"{coeff:+}x_{{{j + 1}}}" if coeff >= 0 else f"{coeff}x_{{{j + 1}}}"
                      if abs(coeff) == 1:
                           term = f"+x_{{{j + 1}}}" if coeff == 1 else f"-x_{{{j + 1}}}"
                      # Adjust sign for the first term in constraint
                      if not constraint_terms and coeff > 0: # First positive term
                           term = term.lstrip('+')
                      constraint_terms.append(term)

        constraint_str = ' '.join(constraint_terms) if constraint_terms else "0"
        if i < len(b):  # Check bounds before accessing b
            latex += f"\qquad {constraint_str} {constraint_symbol} {b[i]}"
        else:
            latex += f"\qquad {constraint_str} {constraint_symbol} ?"  # Placeholder if b index is out of bounds
        
        if i < m - 1: latex += r" \\ & " # Add alignment for next row

    latex += r" \\ & \qquad x_j \geq 0 \quad \forall j \in \{1.." + str(n) + r"\}"
    latex += r"\end{align*}"
    return latex


def display_matrix(matrix, name):
    """Display a matrix/vector in a more readable format"""
    st.write(f"{name}:")
    st.write(matrix)


def validate_inputs(c, A, b):
    """Validate input dimensions and values"""
    try:
        if not isinstance(c, np.ndarray) or c.ndim != 1: return False, "c must be a 1D array"
        if not isinstance(A, np.ndarray) or A.ndim != 2: return False, "A must be a 2D array"
        if not isinstance(b, np.ndarray) or b.ndim != 1: return False, "b must be a 1D array"
        return True, "Valid inputs"
    except Exception as e: return False, f"Input validation error: {e}"


def display_iteration(iteration, tableau, solver, use_fractions, fraction_digits=3, pivot_info=None):
    """Display the tableau for a given iteration with proper fraction digit limiting"""
    st.write(f"**Iteration {iteration}**")

    # Display pivot information if available
    if pivot_info:
        entering_var, leaving_row = pivot_info
        st.write(f"Pivot: Entering Variable = $x_{entering_var + 1}$, Leaving Row = $R_{leaving_row}$")

    # Create DataFrame for display
    headers = [""] + [f"x{i + 1}" for i in range(solver.n)] + \
              [f"s{i + 1}" for i in range(solver.m)] + ["RHS"]

    # Format tableau data with proper fraction digit limiting
    def format_value(x, is_z_row=False):
        if use_fractions:
            # Convert to Fraction
            frac = solver._limit_fraction(Fraction(float(x)))
            # Get numerator and denominator
            num, den = frac.numerator, frac.denominator
            # For z row, don't limit fraction digits
            if is_z_row:
                return str(frac)
            # For other rows, limit based on fraction_digits
            max_value = 10 ** fraction_digits
            if abs(num) > max_value or den > max_value:
                return f"{float(x):.{fraction_digits}f}"
            return str(frac)
        else:
            return f"{float(x):.4f}"

    # Format tableau with special handling for z row
    formatted_data = []
    for i in range(tableau.shape[0]):
        row = []
        for j in range(tableau.shape[1]):
            is_z_row = (i == 0)  # Check if this is the z row
            row.append(format_value(tableau[i, j], is_z_row))
        formatted_data.append(row)

    # Create rows with labels
    rows = []
    rows.append(["z"] + formatted_data[0])
    for i in range(1, solver.m + 1):
        rows.append([f"R{i}"] + formatted_data[i])

    df = pd.DataFrame(rows, columns=headers)

    # Style the DataFrame
    styled_df = df.style.set_properties(**{
        'text-align': 'right',
        'font-family': 'monospace',
        'padding': '5px 10px',
        'color': '#000000',
        'background-color': '#f8f9fa',
        'border': '1px solid #dee2e6'
    })

    st.write(styled_df)

    # Display basis variables
    latex_basis = []
    for i in range(solver.n + solver.m):
        col = tableau[:, i]
        if np.sum(col == 1) == 1 and np.sum(col == 0) == solver.m:
            if i < solver.n:
                latex_basis.append(f"x_{i + 1}")
            else:
                latex_basis.append(f"s_{i + 1 - solver.n}")

    basis_latex = "\\text{Current Basis: } \\mathcal{B} = \\{" + ", ".join(latex_basis) + "\\}"
    st.latex(basis_latex)


def display_solution(solver, use_fractions):
    """Display the solution in either fraction or decimal format"""
    storage = st.session_state.solution_storage

    solution_latex = "\\begin{align*}\n"

    if use_fractions:
        # Convert solution vector to fractions
        fraction_solution = [
            str(solver._limit_fraction(Fraction(float(val))))
            for val in storage.decimal_solution
        ]
        # Convert z* to fraction without digit limiting
        z_optimal = str(solver._limit_fraction(Fraction(float(storage.decimal_optimal))))

        solution_vector = " \\\\ ".join(fraction_solution)
        solution_latex += f"\\mathbf{{x}}^* = \\begin{{pmatrix}} {solution_vector} \\end{{pmatrix}} & \\qquad & z^* = {z_optimal}"
    else:
        solution_vector = " \\\\ ".join([f"{val:.4f}" for val in storage.decimal_solution])
        solution_latex += f"\\mathbf{{x}}^* = \\begin{{pmatrix}} {solution_vector} \\end{{pmatrix}} & \\qquad & z^* = {storage.decimal_optimal:.4f}"

    solution_latex += "\\end{align*}"
    st.latex(solution_latex)


def plot_lp_problem(c, A, b, solution, path_vertices=None, min_x=-1, max_x=10, min_y=-1, max_y=10):
    """
    Plot the LP problem, feasible region, optimal solution, and simplex path in 2D.
    
    Args:
        c: objective coefficients
        A: constraint matrix
        b: right-hand side
        solution: optimal solution (first 2 values used)
        path_vertices: list of points representing the simplex path [optional]
        min_x, max_x, min_y, max_y: initial bounds for plot (will be adjusted)
    
    Returns:
        Matplotlib figure or None, error message if any
    """
    try:
        # Make sure input dimensions are correct
        if len(c) < 2 or A.shape[1] < 2:
            return None, "Cannot visualize problem: need at least 2 variables for 2D plot"

        # Use only the first 2 coefficients and columns for 2D plot
        c_2d = c[:2]
        A_2d = A[:, :2]

        # --- Determine Plot Bounds ---
        # Start with initial bounds or defaults
        plot_min_x, plot_max_x = min_x, max_x
        plot_min_y, plot_max_y = min_y, max_y

        x_vals = np.array([])
        y_vals = np.array([])

        if solution is not None and len(solution) >= 2:
            x_vals = np.append(x_vals, solution[0])
            y_vals = np.append(y_vals, solution[1])

        if path_vertices and len(path_vertices) > 0:
            for vertex in path_vertices:
                if len(vertex) >= 2:
                    x_vals = np.append(x_vals, vertex[0])
                    y_vals = np.append(y_vals, vertex[1])

        # Add origin to bounds consideration
        x_vals = np.append(x_vals, 0)
        y_vals = np.append(y_vals, 0)

        # Calculate automatic bounds based on solution and path if available
        if len(x_vals) > 0 and len(y_vals) > 0:
            sol_min_x, sol_max_x = np.min(x_vals), np.max(x_vals)
            sol_min_y, sol_max_y = np.min(y_vals), np.max(y_vals)

            # Create margin around solution/path points (at least 20%, minimum margin of 2)
            x_margin = max(2, (sol_max_x - sol_min_x) * 0.5) if sol_max_x > sol_min_x else 2
            y_margin = max(2, (sol_max_y - sol_min_y) * 0.5) if sol_max_y > sol_min_y else 2

            # Update bounds to include solution/path with margin, ensuring non-negativity if needed
            # Ensure plot starts at or before 0 (or initial min_x)
            plot_min_x = min(plot_min_x, sol_min_x - x_margin)
            plot_max_x = max(plot_max_x, sol_max_x + x_margin)
            plot_min_y = min(plot_min_y, sol_min_y - y_margin)
            plot_max_y = max(plot_max_y, sol_max_y + y_margin)
        else:
            # If no solution/path, ensure bounds include origin
            plot_min_x = min(plot_min_x, -1)
            plot_max_x = max(plot_max_x, 10)
            plot_min_y = min(plot_min_y, -1)
            plot_max_y = max(plot_max_y, 10)

        # Ensure minimum plot range
        if plot_max_x - plot_min_x < 1: plot_max_x = plot_min_x + 1
        if plot_max_y - plot_min_y < 1: plot_max_y = plot_min_y + 1

        # --- Create Plot ---
        fig, ax = plt.subplots(figsize=(8, 6))

        # Generate grid of points based on final bounds
        x_grid = np.linspace(plot_min_x, plot_max_x, 400) # Increased resolution slightly
        y_grid = np.linspace(plot_min_y, plot_max_y, 400)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Plot constraints
        constraint_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
        for i in range(A_2d.shape[0]):
            # Skip empty constraints (all zeros)
            if np.all(np.abs(A_2d[i]) < 1e-10):
                continue

            a1, a2 = A_2d[i, 0], A_2d[i, 1]
            bi = b[i]
            label = f'C{i+1}: {a1:.2g}x₁ + {a2:.2g}x₂ ≤ {bi:.2g}'
            color = constraint_colors[i % len(constraint_colors)]

            # Plot the line for this constraint within the plot bounds
            if abs(a2) > 1e-9:  # y coefficient non-zero, plot y = (b - a1*x) / a2
                constraint_y = (bi - a1 * x_grid) / a2
                ax.plot(x_grid, constraint_y, color=color, label=label)
            elif abs(a1) > 1e-9:  # x coefficient non-zero, plot x = b / a1 (vertical)
                constraint_x = bi / a1
                ax.axvline(x=constraint_x, color=color, label=label)
            # else: constraint is 0 <= bi, which is either always true or always false

        # Plot feasible region
        mask = np.ones(X.shape, dtype=bool)
        for i in range(A_2d.shape[0]):
            if np.all(np.abs(A_2d[i]) < 1e-10): continue # Skip zero constraints
            # Add small tolerance for boundary checks
            mask = mask & (A_2d[i, 0] * X + A_2d[i, 1] * Y <= b[i] + 1e-9)

        # Also mask non-negative constraints for x and y
        mask = mask & (X >= -1e-9) & (Y >= -1e-9) # Use small tolerance

        # Fill feasible region
        ax.imshow(np.where(mask, 0.1, np.nan), extent=[plot_min_x, plot_max_x, plot_min_y, plot_max_y],
                 origin='lower', cmap='Blues', alpha=0.3, aspect='auto') # Use aspect='auto'

        # --- Plot objective function contours --- ### MODIFICATION START ###
        if not np.all(np.abs(c_2d) < 1e-10):
            Z = c_2d[0] * X + c_2d[1] * Y # Calculate objective value over the grid

            # Determine the range of Z within the plot bounds
            z_min_plot = np.min(Z)
            z_max_plot = np.max(Z)

            # Ensure a small range if min/max are very close or equal
            if np.isclose(z_min_plot, z_max_plot):
                 z_max_plot = z_min_plot + max(abs(z_min_plot * 0.1), 1.0) # Add buffer
                 if np.isclose(z_min_plot, z_max_plot): # If still close (e.g., near zero)
                     z_max_plot = z_min_plot + 1.0

            # Create levels within the calculated range (ensure increasing)
            num_levels = 10
            try:
                # linspace might still fail if min/max are inf/nan, though unlikely here
                z_levels = np.linspace(z_min_plot, z_max_plot, num_levels)
                # Ensure levels are unique in case of numerical issues
                z_levels = np.unique(np.round(z_levels, 8))
                if len(z_levels) < 2: # Need at least 2 levels for contour
                     # Fallback: let contour choose
                     CS = ax.contour(X, Y, Z, num_levels, colors='gray', linestyles='dashed', alpha=0.7)
                else:
                     CS = ax.contour(X, Y, Z, levels=z_levels, colors='gray', linestyles='dashed', alpha=0.7)

                ax.clabel(CS, inline=True, fontsize=8, fmt='%.1f')

                # Add direction of improvement arrow (for MINIMIZATION)
                # Arrow points in the direction of decreasing Z
                mid_x, mid_y = (plot_min_x + plot_max_x) / 2, (plot_min_y + plot_max_y) / 2
                arrow_len = min((plot_max_x - plot_min_x), (plot_max_y - plot_min_y)) / 10
                # Point opposite to the gradient vector c = [c1, c2]
                ax.arrow(mid_x, mid_y, -c_2d[0] * arrow_len, -c_2d[1] * arrow_len,
                        head_width=arrow_len/3, head_length=arrow_len/2, fc='black', ec='black',
                        label='Improvement Direction (Min)')

            except ValueError as ve:
                 return None, f"Error creating contours: {ve}. Min/Max Z: {z_min_plot}/{z_max_plot}"
        # --- ### MODIFICATION END ### ---

        # Plot simplex path if provided
        if path_vertices and len(path_vertices) > 0:
            path_x = [v[0] for v in path_vertices if len(v) >= 2]
            path_y = [v[1] for v in path_vertices if len(v) >= 2]

            if len(path_x) > 1:  # Need at least 2 points for a path
                ax.plot(path_x, path_y, 'o-', color='purple', markersize=5, linewidth=1.5,
                       label='Simplex Path', zorder=5)
                for i, (px, py) in enumerate(zip(path_x, path_y)):
                    offset_x = (plot_max_x - plot_min_x) * 0.015 # Slightly larger offset
                    offset_y = (plot_max_y - plot_min_y) * 0.015
                    ax.annotate(f'{i}', xy=(px, py), xytext=(px + offset_x, py + offset_y),
                               fontsize=8, color='purple', fontweight='bold', zorder=6)

        # Plot optimal solution
        if solution is not None and len(solution) >= 2:
             sol_x, sol_y = solution[0], solution[1]
             # Check if solution is within reasonable bounds of the plot
             if plot_min_x <= sol_x <= plot_max_x and plot_min_y <= sol_y <= plot_max_y:
                 ax.scatter(sol_x, sol_y, color='green', s=100, marker='*',
                           label=f'Optimal: ({sol_x:.3g}, {sol_y:.3g})', zorder=10)
             else:
                 # Indicate solution exists but is outside the current view
                 ax.text(0.95, 0.01, f'Optimal: ({sol_x:.3g}, {sol_y:.3g}) (Outside View)',
                         verticalalignment='bottom', horizontalalignment='right',
                         transform=ax.transAxes, color='green', fontsize=8, style='italic')


        # Configure plot
        ax.set_xlim(plot_min_x, plot_max_x)
        ax.set_ylim(plot_min_y, plot_max_y)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')

        # Adjust title based on minimization objective
        ax.set_title(f'LP Problem: Minimize {c_2d[0]:.2g}x₁ + {c_2d[1]:.2g}x₂')

        # Show the legend without duplicate entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # Place legend outside plot area if too crowded
        if len(by_label) > 5:
             ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        else:
             ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)

        plt.tight_layout(rect=[0, 0, 0.85 if len(by_label) > 5 else 1, 1]) # Adjust layout if legend is outside
        return fig, None

    except Exception as e:
        import traceback
        print(traceback.format_exc()) # Print full traceback to console for debugging
        return None, f"Error creating 2D plot: {str(e)}"


def add_visualization_to_main(c, A, b, solution):
    """
    Add visualization with advanced controls to the Streamlit app.

    Parameters:
    -----------
    c : array-like
        Coefficients of the objective function
    A : array-like
        Constraint coefficients matrix
    b : array-like
        Right-hand side of constraints
    solution : array-like
        The optimal solution
    """
    import streamlit as st

    st.header("Problem Visualization")

    # Determine if this is a large-scale problem
    is_large_scale = False
    if solution is not None:
        if solution[0] > 100 or solution[1] > 100:
            is_large_scale = True

    if is_large_scale:
        st.info("This problem has a large-scale solution. You can choose how to visualize it.")

        viz_option = st.radio(
            "Visualization Option:",
            ["Show feasible region (may not show optimal point)",
             "Show full problem (may be zoomed out)",
             "Focus around solution"],
            index=0
        )

        if viz_option == "Show feasible region (may not show optimal point)":
            # Default visualization (may not show the optimal point)
            fig = plot_lp_problem(c, A, b, solution, "LP Problem Visualization")
            st.pyplot(fig)

            # Add note about the solution
            st.write(
                f"**Note:** The optimal solution at ({solution[0]:.2f}, {solution[1]:.2f}) may be outside the visible area.")

        elif viz_option == "Show full problem (may be zoomed out)":
            # Custom visualization to show entire problem (very zoomed out)
            fig = plot_full_problem(c, A, b, solution)
            st.pyplot(fig)

        else:  # Focus around solution
            # Custom visualization focused on the solution area
            fig = plot_solution_focus(c, A, b, solution)
            st.pyplot(fig)
    else:
        # Standard visualization for normal-scale problems
        fig = plot_lp_problem(c, A, b, solution, "LP Problem Visualization")
        st.pyplot(fig)


def plot_full_problem(c, A, b, solution):
    """
    Create a visualization showing the entire problem no matter how large.

    This might be very zoomed out for large problems.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(10, 8))

    # Set bounds to fully contain the solution and feasible region
    x_max = max(200, solution[0] * 1.2)
    y_max = max(200, solution[1] * 1.2)

    # Draw constraints
    constraint_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for i in range(len(b)):
        a1, a2 = A[i]
        if a1 == 0 and a2 == 0:
            continue

        color = constraint_colors[i % len(constraint_colors)]

        if a2 == 0:  # Vertical line
            x_val = b[i] / a1
            if 0 <= x_val <= x_max * 1.1:
                ax.axvline(x=x_val, color=color, linewidth=2.5,
                           label=f'Constraint {i + 1}: {a1}x₁ + {a2}x₂ ≤ {b[i]}')
        elif a1 == 0:  # Horizontal line
            y_val = b[i] / a2
            if 0 <= y_val <= y_max * 1.1:
                ax.axhline(y=y_val, color=color, linewidth=2.5,
                           label=f'Constraint {i + 1}: {a1}x₁ + {a2}x₂ ≤ {b[i]}')
        else:  # Regular line
            # Calculate endpoints that span the entire visible area
            x_vals = [0, x_max]
            y_vals = [(b[i] - a1 * x) / a2 for x in x_vals]

            # Filter out points with negative y
            valid_points = [(x, y) for x, y in zip(x_vals, y_vals) if y >= 0]

            # If we lost points due to negative y, add points at y=0
            if len(valid_points) < 2:
                x_at_y0 = b[i] / a1
                if 0 <= x_at_y0 <= x_max:
                    valid_points.append((x_at_y0, 0))

            # If we have at least 2 valid points, plot the line
            if len(valid_points) >= 2:
                xs, ys = zip(*valid_points)
                ax.plot(xs, ys, color=color, linewidth=2.5,
                        label=f'Constraint {i + 1}: {a1}x₁ + {a2}x₂ ≤ {b[i]}')

    # Draw non-negativity constraints
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2.5, label='x₁ ≥ 0')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2.5, label='x₂ ≥ 0')

    # Mark the optimal solution
    ax.scatter(solution[0], solution[1], color='red', s=150, marker='*',
               label=f'Optimal Solution ({solution[0]:.2f}, {solution[1]:.2f})')

    # Calculate objective value
    obj_value = c[0] * solution[0] + c[1] * solution[1]

    # Annotate the solution
    ax.annotate(f'Objective value: {obj_value:.2f}',
                xy=(solution[0], solution[1]),
                xytext=(solution[0] * 0.9, solution[1] * 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    # Set labels and title
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Full Problem Visualization (Zoomed Out)', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add legend with smaller font size
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def plot_solution_focus(c, A, b, solution):
    """
    Create a visualization focused around the optimal solution.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Set bounds to focus around the solution
    margin = 30  # Margin around solution to show
    x_min = max(0, solution[0] - margin)
    y_min = max(0, solution[1] - margin)
    x_max = solution[0] + margin
    y_max = solution[1] + margin

    # Draw constraints
    constraint_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for i in range(len(b)):
        a1, a2 = A[i]
        if a1 == 0 and a2 == 0:
            continue

        color = constraint_colors[i % len(constraint_colors)]

        if a2 == 0:  # Vertical line
            x_val = b[i] / a1
            if x_min <= x_val <= x_max:
                ax.axvline(x=x_val, color=color, linewidth=2.5,
                           label=f'Constraint {i + 1}: {a1}x₁ + {a2}x₂ ≤ {b[i]}')
        elif a1 == 0:  # Horizontal line
            y_val = b[i] / a2
            if y_min <= y_val <= y_max:
                ax.axhline(y=y_val, color=color, linewidth=2.5,
                           label=f'Constraint {i + 1}: {a1}x₁ + {a2}x₂ ≤ {b[i]}')
        else:  # Regular line
            # Calculate line endpoints for the visible area
            x_vals = [x_min, x_max]
            y_vals = [(b[i] - a1 * x) / a2 for x in x_vals]

            # Check if line is visible in the view window
            if any(y_min <= y <= y_max for y in y_vals):
                ax.plot(x_vals, y_vals, color=color, linewidth=2.5,
                        label=f'Constraint {i + 1}: {a1}x₁ + {a2}x₂ ≤ {b[i]}')

    # Draw axis lines if visible
    if x_min <= 0 <= x_max:
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2.5, label='x₁ ≥ 0')
    if y_min <= 0 <= y_max:
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2.5, label='x₂ ≥ 0')

    # Mark the optimal solution
    ax.scatter(solution[0], solution[1], color='red', s=150, marker='*',
               label=f'Optimal Solution ({solution[0]:.2f}, {solution[1]:.2f})')

    # Calculate objective value
    obj_value = c[0] * solution[0] + c[1] * solution[1]

    # Annotate the solution
    ax.annotate(f'Objective value: {obj_value:.2f}',
                xy=(solution[0], solution[1]),
                xytext=(solution[0] - margin * 0.2, solution[1] - margin * 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    # Set labels and title
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Solution Focus Visualization', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add legend with smaller font size
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def what_if_analysis_tab(sensitivity, original_c, original_A, original_b, solver):
    """
    Display the what-if analysis tab with improved re-solve functionality.
    """
    st.subheader("What-If Analysis")

    st.write("""
    Use this tool to explore how changes to multiple parameters simultaneously affect your optimal solution.
    """)

    # Create two columns
    col1, col2 = st.columns(2)

    # Column 1: Objective coefficients
    with col1:
        st.write("**Objective Coefficients**")

        # Create sliders for each objective coefficient
        new_c = original_c.copy()
        obj_sensitivity = sensitivity.objective_sensitivity_analysis()

        for j in range(solver.n):
            if j in obj_sensitivity:
                current_value = original_c[j]
                lower, upper = obj_sensitivity[j]

                # Set reasonable bounds for the slider
                min_val = max(lower, current_value - 5) if lower != -np.inf else current_value - 5
                max_val = min(upper, current_value + 5) if upper != np.inf else current_value + 5

                new_c[j] = st.slider(
                    f"c{j + 1} ({current_value})",
                    float(min_val),
                    float(max_val),
                    float(current_value),
                    step=0.1,
                    key=f"slider_c_{j}"
                )

    # Column 2: RHS values
    with col2:
        st.write("**Right-Hand Side Values**")

        # Create sliders for each RHS value
        new_b = original_b.copy()
        rhs_sensitivity = sensitivity.rhs_sensitivity_analysis()

        for i in range(solver.m):
            if i in rhs_sensitivity:
                current_value = original_b[i]
                lower, upper = rhs_sensitivity[i]

                # Set reasonable bounds for the slider
                min_val = max(lower, current_value - 5) if lower != -np.inf else current_value - 5
                max_val = min(upper, current_value + 5) if upper != np.inf else current_value + 5

                new_b[i] = st.slider(
                    f"b{i + 1} ({current_value})",
                    float(min_val),
                    float(max_val),
                    float(current_value),
                    step=0.1,
                    key=f"slider_b_{i}"
                )

    # Check if any parameter is outside its sensitivity range
    obj_outside_range = False
    for j in range(solver.n):
        if j in obj_sensitivity:
            lower, upper = obj_sensitivity[j]
            if (lower != -np.inf and new_c[j] < lower) or (upper != np.inf and new_c[j] > upper):
                obj_outside_range = True
                break

    rhs_outside_range = False
    for i in range(solver.m):
        if i in rhs_sensitivity:
            lower, upper = rhs_sensitivity[i]
            if (lower != -np.inf and new_b[i] < lower) or (upper != np.inf and new_b[i] > upper):
                rhs_outside_range = True
                break

    if obj_outside_range or rhs_outside_range:
        st.warning(
            "One or more parameters are outside their sensitivity ranges. The optimal basis would change, requiring a re-solve of the problem.")

        # Offer option to re-solve with improved functionality
        if st.button("Re-solve with new parameters", key="resolve_button"):
            # Store the new parameters in session state
            st.session_state.sensitivity_resample = {
                'c': new_c.copy(),  # Make sure to use copy to avoid reference issues
                'A': original_A.copy(),
                'b': new_b.copy()
            }
            st.session_state.auto_solve = True  # Flag to trigger automatic solving
            st.rerun()  # Force a rerun of the app
    else:
        # Calculate new objective value using sensitivity information
        shadow_prices = sensitivity.shadow_prices()
        delta_obj = 0

        # Effect of RHS changes
        for i in range(solver.m):
            delta_b = new_b[i] - original_b[i]
            delta_obj += shadow_prices[i] * delta_b

        # Effect of objective coefficient changes
        for j in range(solver.n):
            if sensitivity.optimal_solution[j] > 0:  # Only for basic variables with positive values
                delta_c = new_c[j] - original_c[j]
                delta_obj += delta_c * sensitivity.optimal_solution[j]

        new_obj = sensitivity.optimal_obj + delta_obj

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Original Objective Value",
                f"{sensitivity.optimal_obj:.4f}"
            )
        with col2:
            st.metric(
                "New Objective Value",
                f"{new_obj:.4f}",
                f"{new_obj - sensitivity.optimal_obj:.4f}"
            )

        # If it's a 2D problem, visualize the combined effect
        if solver.n == 2:
            st.subheader("Visualization of Changes")

            fig = plot_combined_sensitivity(
                solver, original_c, new_c, original_A, original_b, new_b,
                sensitivity.optimal_solution
            )
            st.plotly_chart(fig, use_container_width=True)


def display_sensitivity_analysis(solver, original_c, original_A, original_b, use_fractions=False):
    """
    Display sensitivity analysis results in the Streamlit UI.
    Only show interactive plots for 2D problems.
    """
    # Import necessary libraries
    from simplex import SensitivityAnalysis
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np

    # Perform sensitivity analysis
    sensitivity = SensitivityAnalysis(solver)
    is_2d = (solver.n == 2) # Check if it's a 2D problem

    st.header("Sensitivity Analysis")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Objective Coefficients",
        "Right-Hand Side",
        "Shadow Prices",
        "What-If Analysis"
    ])

    # Tab 1: Objective Coefficient Sensitivity
    with tab1:
        st.subheader("Objective Function Coefficient Sensitivity")

        obj_sensitivity = sensitivity.objective_sensitivity_analysis()

        # Convert to DataFrame for better display
        obj_data = []
        for j in range(solver.n):
            if j in obj_sensitivity:
                current_value = original_c[j]
                lower, upper = obj_sensitivity[j]

                # Calculate allowable changes
                delta_lower = "Any decrease" if lower == -np.inf else f"{current_value - lower:.4f}"
                delta_upper = "Any increase" if upper == np.inf else f"{upper - current_value:.4f}"

                obj_data.append({
                    "Variable": f"x{j + 1}",
                    "Current Value": f"{current_value:.4f}",
                    "Lower Bound": "-∞" if lower == -np.inf else f"{lower:.4f}",
                    "Upper Bound": "+∞" if upper == np.inf else f"{upper:.4f}",
                    "Allowable Decrease": delta_lower,
                    "Allowable Increase": delta_upper
                })

        if obj_data:
            obj_df = pd.DataFrame(obj_data)
            st.dataframe(obj_df, use_container_width=True)

            # Interactive analysis code
            st.subheader("Interactive Objective Coefficient Analysis")
            
            if not is_2d:
                st.info("Interactive visualization is only available for 2D problems.")

            # Select a variable to analyze
            selected_var = st.selectbox(
                "Select variable to analyze:",
                [f"x{j + 1}" for j in range(solver.n)],
                key="obj_tab_select"
            )
            j = int(selected_var[1:]) - 1

            if j in obj_sensitivity:
                current_value = original_c[j]
                lower, upper = obj_sensitivity[j]

                # Create a slider for adjusting the coefficient
                min_val = max(lower, current_value - 10) if lower != -np.inf else current_value - 10
                max_val = min(upper, current_value + 10) if upper != np.inf else current_value + 10

                new_value = st.slider(
                    f"Adjusted coefficient for {selected_var}:",
                    float(min_val),
                    float(max_val),
                    float(current_value),
                    step=0.1,
                    key="obj_tab_slider"
                )

                # Show effect on objective value if within allowable range
                is_within_range = (lower == -np.inf or new_value >= lower) and (upper == np.inf or new_value <= upper)

                if is_within_range:
                    # Calculate the new objective value
                    delta_c = new_value - current_value
                    # Ensure optimal_solution index exists (safe for n>=1)
                    sol_j = sensitivity.optimal_solution[j] if j < len(sensitivity.optimal_solution) else 0
                    new_obj = sensitivity.optimal_obj + delta_c * sol_j

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Original Objective Value",
                            f"{sensitivity.optimal_obj:.4f}"
                        )
                    with col2:
                        st.metric(
                            "New Objective Value",
                            f"{new_obj:.4f}",
                            f"{new_obj - sensitivity.optimal_obj:.4f}"
                        )

                    # Only visualize for 2D problems
                    if is_2d:
                        st.subheader("Effect on Objective Function (2D)")

                        # Calculate new objective function
                        new_c = original_c.copy()
                        new_c[j] = new_value

                        # Create a visualization
                        fig = plot_objective_sensitivity(
                            solver, original_c, new_c,
                            sensitivity.optimal_solution,
                            j, sensitivity.optimal_obj, new_obj
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("The selected value is outside the allowable range. The optimal basis would change.")
            else:
                st.info("No sensitivity information available for this variable.")

    # Tab 2: Right-Hand Side Sensitivity
    with tab2:
        st.subheader("Right-Hand Side Sensitivity")

        rhs_sensitivity = sensitivity.rhs_sensitivity_analysis()

        # Convert to DataFrame for better display
        rhs_data = []
        for i in range(solver.m):
            if i in rhs_sensitivity:
                current_value = original_b[i]
                lower, upper = rhs_sensitivity[i]

                # Calculate allowable changes
                delta_lower = "Any decrease" if lower == -np.inf else f"{current_value - lower:.4f}"
                delta_upper = "Any increase" if upper == np.inf else f"{upper - current_value:.4f}"

                rhs_data.append({
                    "Constraint": f"Constraint {i + 1}",
                    "Current RHS": f"{current_value:.4f}",
                    "Lower Bound": "-∞" if lower == -np.inf else f"{lower:.4f}",
                    "Upper Bound": "+∞" if upper == np.inf else f"{upper:.4f}",
                    "Allowable Decrease": delta_lower,
                    "Allowable Increase": delta_upper
                })

        if rhs_data:
            rhs_df = pd.DataFrame(rhs_data)
            st.dataframe(rhs_df, use_container_width=True)

            # Create an interactive visualization
            st.subheader("Interactive RHS Analysis")
            
            if not is_2d:
                st.info("Interactive visualization is only available for 2D problems.")

            # Select a constraint to analyze
            selected_constraint = st.selectbox(
                "Select constraint to analyze:",
                [f"Constraint {i + 1}" for i in range(solver.m)],
                key="rhs_tab_select"
            )
            i = int(selected_constraint.split()[-1]) - 1

            if i in rhs_sensitivity:
                current_value = original_b[i]
                lower, upper = rhs_sensitivity[i]

                # Create a slider for adjusting the RHS
                min_val = max(lower, current_value - 10) if lower != -np.inf else current_value - 10
                max_val = min(upper, current_value + 10) if upper != np.inf else current_value + 10

                new_value = st.slider(
                    f"Adjusted RHS for {selected_constraint}:",
                    float(min_val),
                    float(max_val),
                    float(current_value),
                    step=0.1,
                    key="rhs_tab_slider"
                )

                # Show effect on objective value if within allowable range
                is_within_range = (lower == -np.inf or new_value >= lower) and (upper == np.inf or new_value <= upper)

                if is_within_range:
                    # Get shadow price for this constraint
                    shadow_prices = sensitivity.shadow_prices()
                    shadow_price = shadow_prices[i] if i < len(shadow_prices) else 0

                    # Calculate the new objective value
                    delta_b = new_value - current_value
                    new_obj = sensitivity.optimal_obj + shadow_price * delta_b

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Original Objective Value",
                            f"{sensitivity.optimal_obj:.4f}"
                        )
                    with col2:
                        st.metric(
                            "New Objective Value",
                            f"{new_obj:.4f}",
                            f"{new_obj - sensitivity.optimal_obj:.4f}"
                        )

                    # Only visualize for 2D problems
                    if is_2d:
                        st.subheader("Effect on Feasible Region (2D)")

                        # Create a new b vector
                        new_b = original_b.copy()
                        new_b[i] = new_value

                        # Create a visualization
                        fig = plot_rhs_sensitivity(
                            solver, original_A, original_b, new_b,
                            sensitivity.optimal_solution, i
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("The selected value is outside the allowable range. The optimal basis would change.")
            else:
                st.info("No sensitivity information available for this constraint.")

    # Tab 3: Shadow Prices
    with tab3:
        st.subheader("Shadow Prices (Dual Values)")

        shadow_prices = sensitivity.shadow_prices()

        # Convert to DataFrame for better display
        shadow_data = []
        for i in range(solver.m):
            price = shadow_prices[i] if i < len(shadow_prices) else 0
            shadow_data.append({
                "Constraint": f"Constraint {i + 1}",
                "Shadow Price": f"{price:.4f}",
                "Interpretation": "Change in objective per unit increase in RHS"
            })

        if shadow_data:
            shadow_df = pd.DataFrame(shadow_data)
            st.dataframe(shadow_df, use_container_width=True)

            # Create a bar chart of shadow prices
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"Constraint {i + 1}" for i in range(solver.m)],
                    y=shadow_prices,
                    marker_color='royalblue'
                )
            ])
            fig.update_layout(
                title="Shadow Prices by Constraint",
                xaxis_title="Constraint",
                yaxis_title="Shadow Price",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write("""
            **What are Shadow Prices?**

            Shadow prices represent the rate of change in the objective function value per unit change in the right-hand side of the constraint.

            - A **positive** shadow price means that increasing the resource (right-hand side) will increase the objective value
            - A **negative** shadow price means that increasing the resource will decrease the objective value
            - A **zero** shadow price indicates that the constraint is not binding at the optimal solution
            """)

    # Tab 4: What-If Analysis
    with tab4:
        st.subheader("What-If Analysis")

        st.write("""
        Use this tool to explore how changes to multiple parameters simultaneously affect your optimal solution.
        """)

        if not is_2d:
            st.info("Interactive visualization is only available for 2D problems.")

        # Create two columns
        col1, col2 = st.columns(2)

        # Column 1: Objective coefficients
        with col1:
            st.write("**Objective Coefficients**")

            # Create sliders for each objective coefficient
            new_c = original_c.copy()
            obj_sensitivity = sensitivity.objective_sensitivity_analysis()

            for j in range(solver.n):
                if j in obj_sensitivity:
                    current_value = original_c[j]
                    lower, upper = obj_sensitivity[j]

                    # Set reasonable bounds for the slider
                    min_val = max(lower, current_value - 5) if lower != -np.inf else current_value - 5
                    max_val = min(upper, current_value + 5) if upper != np.inf else current_value + 5

                    new_c[j] = st.slider(
                        f"c{j + 1} ({current_value})",
                        float(min_val),
                        float(max_val),
                        float(current_value),
                        step=0.1,
                        key=f"whatif_slider_c_{j}"
                    )

        # Column 2: RHS values
        with col2:
            st.write("**Right-Hand Side Values**")

            # Create sliders for each RHS value
            new_b = original_b.copy()
            rhs_sensitivity = sensitivity.rhs_sensitivity_analysis()

            for i in range(solver.m):
                if i in rhs_sensitivity:
                    current_value = original_b[i]
                    lower, upper = rhs_sensitivity[i]

                    # Set reasonable bounds for the slider
                    min_val = max(lower, current_value - 5) if lower != -np.inf else current_value - 5
                    max_val = min(upper, current_value + 5) if upper != np.inf else current_value + 5

                    new_b[i] = st.slider(
                        f"b{i + 1} ({current_value})",
                        float(min_val),
                        float(max_val),
                        float(current_value),
                        step=0.1,
                        key=f"whatif_slider_b_{i}"
                    )

        # Check if any parameter is outside its sensitivity range
        obj_outside_range = False
        for j in range(solver.n):
            if j in obj_sensitivity:
                lower, upper = obj_sensitivity[j]
                if (lower != -np.inf and new_c[j] < lower) or (upper != np.inf and new_c[j] > upper):
                    obj_outside_range = True
                    break

        rhs_outside_range = False
        for i in range(solver.m):
            if i in rhs_sensitivity:
                lower, upper = rhs_sensitivity[i]
                if (lower != -np.inf and new_b[i] < lower) or (upper != np.inf and new_b[i] > upper):
                    rhs_outside_range = True
                    break

        if obj_outside_range or rhs_outside_range:
            st.warning(
                "One or more parameters are outside their sensitivity ranges. The optimal basis would change, requiring a re-solve of the problem.")

            # Offer option to re-solve with improved tab preservation
            if st.button("Re-solve with new parameters", key="resolve_button"):
                # Store the new parameters in session state
                st.session_state.sensitivity_resample = {
                    'c': new_c.copy(),
                    'A': original_A.copy(),
                    'b': new_b.copy()
                }
                st.session_state.auto_solve = True
                st.session_state.active_tab = 3  # Set to What-If Analysis tab
                st.session_state.coming_from_whatif = True
                st.rerun()
        else:
            # Calculate new objective value using sensitivity information
            shadow_prices = sensitivity.shadow_prices()
            delta_obj = 0

            # Effect of RHS changes
            for i in range(solver.m):
                delta_b = new_b[i] - original_b[i]
                price = shadow_prices[i] if i < len(shadow_prices) else 0
                delta_obj += price * delta_b

            # Effect of objective coefficient changes
            for j in range(solver.n):
                sol_j = sensitivity.optimal_solution[j] if j < len(sensitivity.optimal_solution) else 0
                if sol_j > 1e-6:  # Only for basic variables with positive values
                    delta_c = new_c[j] - original_c[j]
                    delta_obj += delta_c * sol_j

            new_obj = sensitivity.optimal_obj + delta_obj

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Original Objective Value",
                    f"{sensitivity.optimal_obj:.4f}"
                )
            with col2:
                st.metric(
                    "New Objective Value",
                    f"{new_obj:.4f}",
                    f"{new_obj - sensitivity.optimal_obj:.4f}"
                )

            # Only visualize for 2D problems
            if is_2d:
                st.subheader("Visualization of Changes (2D)")

                fig = plot_combined_sensitivity(
                    solver, original_c, new_c, original_A, original_b, new_b,
                    sensitivity.optimal_solution
                )
                st.plotly_chart(fig, use_container_width=True)

    # Track which tab is selected for next time
    if tab4.selected:
        st.session_state.active_tab = 3
    elif tab3.selected:
        st.session_state.active_tab = 2
    elif tab2.selected:
        st.session_state.active_tab = 1
    elif tab1.selected:
        st.session_state.active_tab = 0


def plot_objective_sensitivity(solver, original_c, new_c, optimal_solution, changed_index, original_obj, new_obj):
    """
    Create a visualization showing how changing an objective coefficient affects the problem.

    Parameters:
    -----------
    solver : PrimalSimplex
        The solved simplex instance.
    original_c : array-like
        Original objective function coefficients.
    new_c : array-like
        New objective function coefficients.
    optimal_solution : array-like
        Optimal solution vector.
    changed_index : int
        Index of the coefficient that was changed.
    original_obj : float
        Original optimal objective value.
    new_obj : float
        New optimal objective value.

    Returns:
    --------
    plotly.graph_objects.Figure: The visualization figure.
    """
    # Create a grid of points
    x = np.linspace(0, max(20, optimal_solution[0] * 2), 100)
    y = np.linspace(0, max(20, optimal_solution[1] * 2), 100)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values for original and new objective functions
    Z_original = original_c[0] * X + original_c[1] * Y
    Z_new = new_c[0] * X + new_c[1] * Y

    # Create subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Original Objective", "New Objective"],
                        shared_yaxes=True)

    # Add contour plots for both objective functions
    fig.add_trace(
        go.Contour(
            z=Z_original,
            x=x,
            y=y,
            colorscale='Blues',
            showscale=False,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white')
            )
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Contour(
            z=Z_new,
            x=x,
            y=y,
            colorscale='Reds',
            showscale=False,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white')
            )
        ),
        row=1, col=2
    )

    # Add optimal point to both plots
    fig.add_trace(
        go.Scatter(
            x=[optimal_solution[0]],
            y=[optimal_solution[1]],
            mode='markers',
            marker=dict(color='green', size=10, symbol='star'),
            name='Optimal Solution'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[optimal_solution[0]],
            y=[optimal_solution[1]],
            mode='markers',
            marker=dict(color='green', size=10, symbol='star'),
            name='Optimal Solution'
        ),
        row=1, col=2
    )

    # Add annotations for objective values
    fig.add_annotation(
        x=optimal_solution[0] + 1,
        y=optimal_solution[1] + 1,
        text=f"Obj: {original_obj:.2f}",
        showarrow=True,
        arrowhead=2,
        row=1, col=1
    )

    fig.add_annotation(
        x=optimal_solution[0] + 1,
        y=optimal_solution[1] + 1,
        text=f"Obj: {new_obj:.2f}",
        showarrow=True,
        arrowhead=2,
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title=f"Effect of Changing c{changed_index + 1} from {original_c[changed_index]} to {new_c[changed_index]}",
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # Update axes labels
    fig.update_xaxes(title_text="x₁", row=1, col=1)
    fig.update_xaxes(title_text="x₁", row=1, col=2)
    fig.update_yaxes(title_text="x₂", row=1, col=1)

    return fig


def plot_rhs_sensitivity(solver, A, original_b, new_b, optimal_solution, changed_index):
    """
    Create a visualization showing how changing a RHS value affects the feasible region.

    Parameters:
    -----------
    solver : PrimalSimplex
        The solved simplex instance.
    A : array-like
        Constraint coefficients matrix.
    original_b : array-like
        Original right-hand side values.
    new_b : array-like
        New right-hand side values.
    optimal_solution : array-like
        Optimal solution vector.
    changed_index : int
        Index of the RHS value that was changed.

    Returns:
    --------
    plotly.graph_objects.Figure: The visualization figure.
    """
    # Create a grid of points
    x = np.linspace(0, max(20, optimal_solution[0] * 2), 100)
    y = np.linspace(0, max(20, optimal_solution[1] * 2), 100)
    X, Y = np.meshgrid(x, y)

    # Create figure
    fig = go.Figure()

    # Add constraint lines for original problem
    for i in range(solver.m):
        a, b = A[i, 0], A[i, 1]
        if a == 0:  # Horizontal line
            y_val = original_b[i] / b
            fig.add_trace(
                go.Scatter(
                    x=[0, x[-1]],
                    y=[y_val, y_val],
                    mode='lines',
                    line=dict(color='blue', width=2, dash='dash' if i != changed_index else 'solid'),
                    name=f'Original Constraint {i + 1}'
                )
            )
        elif b == 0:  # Vertical line
            x_val = original_b[i] / a
            fig.add_trace(
                go.Scatter(
                    x=[x_val, x_val],
                    y=[0, y[-1]],
                    mode='lines',
                    line=dict(color='blue', width=2, dash='dash' if i != changed_index else 'solid'),
                    name=f'Original Constraint {i + 1}'
                )
            )
        else:  # General line
            y_vals = (original_b[i] - a * x) / b
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_vals,
                    mode='lines',
                    line=dict(color='blue', width=2, dash='dash' if i != changed_index else 'solid'),
                    name=f'Original Constraint {i + 1}'
                )
            )

    # Add constraint line for changed constraint with new RHS
    i = changed_index
    a, b = A[i, 0], A[i, 1]
    if a == 0:  # Horizontal line
        y_val = new_b[i] / b
        fig.add_trace(
            go.Scatter(
                x=[0, x[-1]],
                y=[y_val, y_val],
                mode='lines',
                line=dict(color='red', width=2),
                name=f'New Constraint {i + 1}'
            )
        )
    elif b == 0:  # Vertical line
        x_val = new_b[i] / a
        fig.add_trace(
            go.Scatter(
                x=[x_val, x_val],
                y=[0, y[-1]],
                mode='lines',
                line=dict(color='red', width=2),
                name=f'New Constraint {i + 1}'
            )
        )
    else:  # General line
        y_vals = (new_b[i] - a * x) / b
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_vals,
                mode='lines',
                line=dict(color='red', width=2),
                name=f'New Constraint {i + 1}'
            )
        )

    # Add optimal point
    fig.add_trace(
        go.Scatter(
            x=[optimal_solution[0]],
            y=[optimal_solution[1]],
            mode='markers',
            marker=dict(color='green', size=10, symbol='star'),
            name='Optimal Solution'
        )
    )

    # Add axis lines
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0, y[-1]],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, x[-1]],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Effect of Changing b{changed_index + 1} from {original_b[changed_index]} to {new_b[changed_index]}",
        xaxis_title="x₁",
        yaxis_title="x₂",
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


def plot_combined_sensitivity(solver, original_c, new_c, A, original_b, new_b, optimal_solution):
    """
    Create a visualization showing combined effect of changing objective and RHS values.

    Parameters:
    -----------
    solver : PrimalSimplex
        The solved simplex instance.
    original_c : array-like
        Original objective function coefficients.
    new_c : array-like
        New objective function coefficients.
    A : array-like
        Constraint coefficients matrix.
    original_b : array-like
        Original right-hand side values.
    new_b : array-like
        New right-hand side values.
    optimal_solution : array-like
        Optimal solution vector.

    Returns:
    --------
    plotly.graph_objects.Figure: The visualization figure.
    """
    # Create a grid of points
    x = np.linspace(0, max(20, optimal_solution[0] * 2), 100)
    y = np.linspace(0, max(20, optimal_solution[1] * 2), 100)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values for original and new objective functions
    Z_original = original_c[0] * X + original_c[1] * Y
    Z_new = new_c[0] * X + new_c[1] * Y

    # Create figure
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Original Problem", "Modified Problem"],
                        shared_yaxes=True)

    # Add contour plots for both objective functions
    fig.add_trace(
        go.Contour(
            z=Z_original,
            x=x,
            y=y,
            colorscale='Blues',
            showscale=False,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white')
            )
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Contour(
            z=Z_new,
            x=x,
            y=y,
            colorscale='Reds',
            showscale=False,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white')
            )
        ),
        row=1, col=2
    )

    # Add constraint lines for original problem
    for i in range(solver.m):
        a, b = A[i, 0], A[i, 1]
        if a == 0:  # Horizontal line
            y_val = original_b[i] / b
            fig.add_trace(
                go.Scatter(
                    x=[0, x[-1]],
                    y=[y_val, y_val],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name=f'Original Constraint {i + 1}'
                ),
                row=1, col=1
            )
        elif b == 0:  # Vertical line
            x_val = original_b[i] / a
            fig.add_trace(
                go.Scatter(
                    x=[x_val, x_val],
                    y=[0, y[-1]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name=f'Original Constraint {i + 1}'
                ),
                row=1, col=1
            )
        else:  # General line
            y_vals = (original_b[i] - a * x) / b
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_vals,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name=f'Original Constraint {i + 1}'
                ),
                row=1, col=1
            )

    # Add constraint lines for new problem
    for i in range(solver.m):
        a, b = A[i, 0], A[i, 1]
        if a == 0:  # Horizontal line
            y_val = new_b[i] / b
            fig.add_trace(
                go.Scatter(
                    x=[0, x[-1]],
                    y=[y_val, y_val],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'New Constraint {i + 1}'
                ),
                row=1, col=2
            )
        elif b == 0:  # Vertical line
            x_val = new_b[i] / a
            fig.add_trace(
                go.Scatter(
                    x=[x_val, x_val],
                    y=[0, y[-1]],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'New Constraint {i + 1}'
                ),
                row=1, col=2
            )
        else:  # General line
            y_vals = (new_b[i] - a * x) / b
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_vals,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'New Constraint {i + 1}'
                ),
                row=1, col=2
            )

    # Add optimal point
    fig.add_trace(
        go.Scatter(
            x=[optimal_solution[0]],
            y=[optimal_solution[1]],
            mode='markers',
            marker=dict(color='green', size=10, symbol='star'),
            name='Optimal Solution'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[optimal_solution[0]],
            y=[optimal_solution[1]],
            mode='markers',
            marker=dict(color='green', size=10, symbol='star'),
            name='Optimal Solution'
        ),
        row=1, col=2
    )

    # Add axis lines to both subplots
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[0, y[-1]],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ),
            row=1, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=[0, x[-1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ),
            row=1, col=col
        )

    # Update layout
    fig.update_layout(
        title="Combined Effect of Parameter Changes",
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    # Update axes labels
    fig.update_xaxes(title_text="x₁", row=1, col=1)
    fig.update_xaxes(title_text="x₁", row=1, col=2)
    fig.update_yaxes(title_text="x₂", row=1, col=1)

    return fig


def plot_lp_problem_3d(c, A, b, solution=None, path_vertices=None, title="3D Linear Programming Visualization"):
    """
    Create a visualization of a 3D linear programming problem, including simplex path.
    """
    # ... (check for 3 variables) ...
    if len(c) < 3 or A.shape[1] < 3:
        st.warning("3D Visualization requires 3 variables.")
        return None

    c = np.array(c[:3], dtype=float) # Ensure use only first 3 coeffs
    A = np.array(A[:, :3], dtype=float) # Ensure use only first 3 cols
    b = np.array(b, dtype=float)


    fig = go.Figure()

    # --- Determine bounds ---
    max_coord = 10.0
    all_coords = []
    if solution is not None and len(solution) >= 3: all_coords.extend(solution[:3])
    if path_vertices:
        for v in path_vertices:
            if len(v) >= 3: all_coords.extend(v[:3])
    # Add axis intercepts
    for i in range(A.shape[0]):
         if abs(A[i,0]) > 1e-9 and b[i]/A[i,0] >= 0: all_coords.append(b[i]/A[i,0])
         if abs(A[i,1]) > 1e-9 and b[i]/A[i,1] >= 0: all_coords.append(b[i]/A[i,1])
         if abs(A[i,2]) > 1e-9 and b[i]/A[i,2] >= 0: all_coords.append(b[i]/A[i,2])

    if all_coords: max_coord = max(max(all_coords) * 1.2, 5.0) # Add buffer, min 5
    else: max_coord = 5.0 # Default if no points

    # --- Grid for Isosurfaces ---
    grid_res = 15 # Resolution
    x_vals = np.linspace(0, max_coord, grid_res)
    y_vals = np.linspace(0, max_coord, grid_res)
    z_vals = np.linspace(0, max_coord, grid_res)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij') # Use 'ij' indexing

    # --- Plot Constraint Planes (as before) ---
    constraint_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(A.shape[0]):
        a1, a2, a3 = A[i, 0], A[i, 1], A[i, 2]
        rhs = b[i]
        constraint_values = a1 * X + a2 * Y + a3 * Z
        label = f'C{i+1}: {a1:.1f}x₁+ {a2:.1f}x₂+ {a3:.1f}x₃ ≤ {rhs:.1f}'
        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=constraint_values.flatten(),
            isomin=rhs, isomax=rhs, surface_count=1,
            opacity=0.3, # Slightly lower opacity
            colorscale=[[0, constraint_colors[i % len(constraint_colors)]], [1, constraint_colors[i % len(constraint_colors)]]],
            showscale=False, name=label
        ))

    # --- Plot Non-negativity Planes (optional, can clutter) ---
    # fig.add_trace(go.Surface(x=[[0,max_coord],[0,max_coord]], y=[[0,0],[0,0]], z=[[0,0],[max_coord,max_coord]], opacity=0.1, colorscale=[[0,'lightgrey'],[1,'lightgrey']], showscale=False, name='x₂=0'))
    # fig.add_trace(go.Surface(x=[[0,0],[0,0]], y=[[0,max_coord],[0,max_coord]], z=[[0,0],[max_coord,max_coord]], opacity=0.1, colorscale=[[0,'lightgrey'],[1,'lightgrey']], showscale=False, name='x₁=0'))
    # fig.add_trace(go.Surface(x=[[0,max_coord],[0,max_coord]], y=[[0,0],[max_coord,max_coord]], z=[[0,0],[0,0]], opacity=0.1, colorscale=[[0,'lightgrey'],[1,'lightgrey']], showscale=False, name='x₃=0'))


    # --- Plot Simplex Path ---
    if path_vertices and len(path_vertices) > 0:
        path_x = [v[0] for v in path_vertices if len(v)>=3]
        path_y = [v[1] for v in path_vertices if len(v)>=3]
        path_z = [v[2] for v in path_vertices if len(v)>=3]
        fig.add_trace(go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            mode='lines+markers',
            line=dict(color='orange', width=4),
            marker=dict(color='orange', size=5),
            name='Simplex Path'
        ))

    # --- Plot Optimal Solution ---
    if solution is not None and len(solution) >= 3:
        x_sol, y_sol, z_sol = solution[0], solution[1], solution[2]
        obj_value = np.dot(c, [x_sol, y_sol, z_sol])
        fig.add_trace(go.Scatter3d(
            x=[x_sol], y=[y_sol], z=[z_sol],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name=f'Optimal ({x_sol:.2f}, {y_sol:.2f}, {z_sol:.2f}), Z={obj_value:.2f}'
        ))

    # --- Configure Layout ---
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x₁', yaxis_title='x₂', zaxis_title='x₃',
            xaxis=dict(range=[0, max_coord], autorange=False, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            yaxis=dict(range=[0, max_coord], autorange=False, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            zaxis=dict(range=[0, max_coord], autorange=False, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            aspectmode='cube',
            camera_eye=dict(x=1.8, y=1.8, z=0.8) # Adjust initial camera angle
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


def initialize_session_state():
    """Initialize all required session state variables"""
    if 'solution_storage' not in st.session_state:
        st.session_state.solution_storage = None
    if 'tableaus' not in st.session_state:
        st.session_state.tableaus = []  # Will store tuples of (iteration, tableau, solver, pivot_info)
    if 'has_solved' not in st.session_state:
        st.session_state.has_solved = False
    if 'sensitivity_resample' not in st.session_state:
        st.session_state.sensitivity_resample = None
    if 'auto_solve' not in st.session_state:
        st.session_state.auto_solve = False


def main():
    # Initialize session state variables
    if 'solution_storage' not in st.session_state:
        st.session_state.solution_storage = None
    if 'tableaus' not in st.session_state:
        st.session_state.tableaus = []  # Will store tuples of (iteration, tableau, solver, pivot_info)
    if 'has_solved' not in st.session_state:
        st.session_state.has_solved = False
    if 'sensitivity_resample' not in st.session_state:
        st.session_state.sensitivity_resample = None
    if 'auto_solve' not in st.session_state:
        st.session_state.auto_solve = False
    if 'solver' not in st.session_state:
        st.session_state.solver = None
    if 'original_params' not in st.session_state:
        st.session_state.original_params = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # Default to first tab
    if 'coming_from_whatif' not in st.session_state:
        st.session_state.coming_from_whatif = False

    st.title("Linear Programming Solver")
    st.write("Solve linear programming problems using the Primal Simplex method")

    # Preserve sidebar state when coming from What-If analysis
    if st.session_state.coming_from_whatif and st.session_state.sensitivity_resample is not None:
        # Force sidebar to be expanded
        st.sidebar.markdown(
            '<script>setTimeout(function() {if (document.getElementsByClassName("css-1adrfps e1fqkh3o2")[0].style.width === "0px") {document.getElementsByClassName("css-1adrfps e1fqkh3o2")[0].style.width="250px";}}, 100);</script>',
            unsafe_allow_html=True
        )
        # Reset the flag after showing sidebar
        st.session_state.coming_from_whatif = False

    # Sidebar for problem setup
    st.sidebar.header("Problem Setup")

    # Option to use example problem
    use_example = st.sidebar.checkbox("Use example problem", value=True, key='use_example')

    # Get problem parameters
    if use_example:
        c, A, b = create_example_problem()
        m, n = A.shape
        # Update sidebar widgets to reflect the example
        st.session_state['m'] = m
        st.session_state['n'] = n
        st.session_state['c_input'] = ",".join(map(str, c))
        st.session_state['A_input'] = "\n".join(",".join(map(str, row)) for row in A)
        st.session_state['b_input'] = ",".join(map(str, b))
    elif st.session_state.sensitivity_resample is not None:
        # Use the resampled parameters from sensitivity analysis
        c = st.session_state.sensitivity_resample['c']
        A = st.session_state.sensitivity_resample['A']
        b = st.session_state.sensitivity_resample['b']
        m, n = A.shape

        st.info("Using modified parameters from sensitivity analysis")
        # Update sidebar widgets
        st.session_state['m'] = m
        st.session_state['n'] = n
        st.session_state['c_input'] = ",".join(map(str, c))
        st.session_state['A_input'] = "\n".join(",".join(map(str, row)) for row in A)
        st.session_state['b_input'] = ",".join(map(str, b))
    else:
        # Get dimensions
        col1, col2 = st.sidebar.columns(2)
        # Use st.session_state values if they exist, otherwise default
        m_val = st.session_state.get('m', 3)
        n_val = st.session_state.get('n', 3) # Default n to 3
        m = col1.number_input("Number of constraints (m)", min_value=1, value=m_val, key='m')
        n = col2.number_input("Number of variables (n)", min_value=1, value=n_val, key='n')

        # Input for objective function coefficients
        st.sidebar.subheader("Objective Function Coefficients (c)")
        c_default = st.session_state.get('c_input', "-2,-3,-4") # Default c for 3D
        c_input = st.sidebar.text_input("Enter c values (comma-separated)", c_default, key='c_input')
        try:
            c = np.array([float(x.strip()) for x in c_input.split(",")])
        except:
            st.error("Invalid input for objective function coefficients")
            return

        # Input for constraint matrix A
        st.sidebar.subheader("Constraint Matrix (A)")
        A_default = st.session_state.get('A_input', "1,1,1\n2,1,0\n0,1,3") # Default A for 3D
        A_input = st.sidebar.text_area("Enter A matrix (one row per line, comma-separated)", A_default, key='A_input')
        try:
            A = np.array([
                [float(x.strip()) for x in row.split(",")]
                for row in A_input.strip().split("\n")
            ])
        except:
            st.error("Invalid input for constraint matrix")
            return

        # Input for RHS values
        st.sidebar.subheader("Right Hand Side (b)")
        b_default = st.session_state.get('b_input', "6,4,7") # Default b for 3D
        b_input = st.sidebar.text_input("Enter b values (comma-separated)", b_default, key='b_input')
        try:
            b = np.array([float(x.strip()) for x in b_input.split(",")])
        except:
            st.error("Invalid input for RHS values")
            return

    # Store current problem parameters
    st.session_state.problem_params = (c, A, b)

    # Fraction display options
    use_fractions = st.sidebar.checkbox("Use fractions", value=True, key='use_fractions')
    fraction_digits = st.sidebar.number_input(
        "Maximum fraction digits",
        min_value=1,
        max_value=5,
        value=3,
        disabled=not use_fractions,
        key='fraction_digits'
    )

    # Add option to use equality constraints
    eq_constraints = st.sidebar.checkbox("Use equality constraints (=) instead of inequalities (≤)",
                                         value=False, key='eq_constraints')

    # Validate inputs
    valid, message = validate_inputs(c, A, b)
    if not valid:
        st.error(message)
        return

    # Display the problem
    st.header("Problem Formulation")
    st.latex(format_lp_problem(c, A, b, n, m, eq_constraints))

    # Solve button or auto-solve from sensitivity analysis
    should_solve = st.button("Solve", key='solve_button') or st.session_state.auto_solve

    if should_solve:
        # Reset auto_solve flag
        st.session_state.auto_solve = False

        # Set solve flag
        st.session_state.has_solved = True

        # Clear previous tableaus when solving new problem
        st.session_state.tableaus = []

        st.header("Solution Process")
        try:
            # Import the sensitivity analysis module
            from simplex import PrimalSimplex, SensitivityAnalysis

            # Create solver instance with fractions
            solver = PrimalSimplex(c.copy(), A.copy(), b.copy(), use_fractions=True, fraction_digits=fraction_digits,
                                   eq_constraints=eq_constraints)

            # Store original parameters for sensitivity analysis
            st.session_state.original_params = {
                'c': c.copy(),
                'A': A.copy(),
                'b': b.copy()
            }

            # Store initial tableau (no pivot info for first tableau)
            iteration = 0
            st.session_state.tableaus.append((iteration, solver.tableau.copy(), solver, None))

            # Solve step by step
            while True:
                pivot_col = solver._find_pivot_column()
                if pivot_col is None:
                    break

                pivot_row = solver._find_pivot_row(pivot_col)
                if pivot_row is None:
                    st.error("Problem is unbounded")
                    return

                # Store pivot information before making the pivot
                pivot_info = (pivot_col, pivot_row)

                # Perform the pivot
                solver._pivot(pivot_row, pivot_col)
                iteration += 1

                # Store tableau with pivot information
                st.session_state.tableaus.append((iteration, solver.tableau.copy(), solver, pivot_info))

            # Extract decimal solution
            decimal_solution = np.zeros(solver.n)
            for i in range(solver.n):
                col = solver.tableau[:, i]
                # Check if it's a basic variable column
                is_basic = (np.sum(col == 1) == 1) and (np.allclose(col[col != 1], 0))
                if is_basic:
                    row_idx = np.where(np.isclose(col, 1))[0][0]
                    # Ensure row_idx is within bounds and not the objective row
                    if row_idx > 0 and row_idx < solver.tableau.shape[0]:
                         # Check if RHS is valid before assignment
                        rhs_val = solver.tableau[row_idx, -1]
                        if not np.isnan(rhs_val) and not np.isinf(rhs_val):
                             decimal_solution[i] = rhs_val

            decimal_optimal = -solver.tableau[0, -1] if not np.isnan(solver.tableau[0, -1]) else np.nan

            # Store only decimal values in solution storage
            st.session_state.solution_storage = SolutionStorage(
                decimal_solution,
                decimal_optimal
            )

            # Store solver for sensitivity analysis
            st.session_state.solver = solver

            # Clear sensitivity_resample after successfully solving
            st.session_state.sensitivity_resample = None

        except Exception as e:
            st.error(f"Error solving problem: {str(e)}")
            # Reset state if solve failed
            st.session_state.has_solved = False
            st.session_state.solution_storage = None
            st.session_state.solver = None
            return

    # Display solution progress if available
    if st.session_state.has_solved and st.session_state.tableaus:
        st.success("Optimal solution found!")

        # Option to show solution progress
        show_progress = st.checkbox("Show solution progress", value=False)

        if show_progress:
            st.markdown("### Solution Progress")
            for iteration, tableau, solver, pivot_info in st.session_state.tableaus:
                display_iteration(iteration, tableau, solver, use_fractions, fraction_digits, pivot_info)

    # Display final solution if available
    if st.session_state.solution_storage is not None and st.session_state.solver is not None:
        st.header("Optimal Solution")
        display_solution(st.session_state.solver, use_fractions)

        # --- Visualization Section ---
        current_n = st.session_state.solver.n # Number of variables from the solver
        solution = st.session_state.solution_storage.decimal_solution
        # Ensure solution has the correct length matching current_n
        if len(solution) != current_n:
            # Handle potential mismatch if solver internals changed n
             st.warning(f"Solution length ({len(solution)}) mismatch with variable count ({current_n}). Using first {current_n} values.")
             solution = solution[:current_n] if len(solution) > current_n else np.pad(solution, (0, current_n - len(solution)))

        # Retrieve potentially modified c, A, b used by the solver if needed
        # Or better, use the original_params if available
        if st.session_state.original_params:
            viz_c = st.session_state.original_params['c']
            viz_A = st.session_state.original_params['A']
            viz_b = st.session_state.original_params['b']
             # Make sure viz_c matches current_n
            if len(viz_c) != current_n:
                 # This case is less likely but handle defensively
                 st.warning("Objective function length mismatch during visualization.")
                 # Fallback or adjust viz_c if necessary
                 viz_c = viz_c[:current_n] if len(viz_c) > current_n else np.pad(viz_c, (0, current_n - len(viz_c)))
        else: # Fallback if original params aren't set
            viz_c, viz_A, viz_b = st.session_state.problem_params

        if current_n == 2:
            st.header("Problem Visualization (2D)")
            # Use existing 2D plotting (including large scale options)
            # Ensure solution has length 2 for 2D plots
            sol_2d = solution[:2] if len(solution) >= 2 else np.pad(solution, (0, 2-len(solution)))
            is_large_scale = sol_2d[0] > 100 or sol_2d[1] > 100
            if is_large_scale:
                st.info("This 2D problem has a large-scale solution. Choose visualization.")
                viz_option = st.radio("Visualization Option:", ["Feasible region", "Full problem", "Focus solution"], index=0)
                if viz_option == "Feasible region":
                    # Use path_vertices=None for the feasible region plot
                    fig, err_msg = plot_lp_problem(viz_c, viz_A, viz_b, sol_2d)
                    if fig: 
                        st.pyplot(fig)
                    elif err_msg:
                        st.warning(err_msg)
                    st.write(f"Note: Optimal solution at ({sol_2d[0]:.2f}, {sol_2d[1]:.2f}) may be outside.")
                elif viz_option == "Full problem":
                    fig = plot_full_problem(viz_c, viz_A, viz_b, sol_2d)
                    if fig: st.pyplot(fig)
                else: # Focus solution
                    fig = plot_solution_focus(viz_c, viz_A, viz_b, sol_2d)
                    if fig: st.pyplot(fig)
            else:
                # Use the simplex path for the regular plot
                simplex_path = getattr(st.session_state.solver, 'path_vertices', None)
                fig, err_msg = plot_lp_problem(viz_c, viz_A, viz_b, sol_2d, path_vertices=simplex_path)
                if fig: 
                    st.pyplot(fig)
                elif err_msg:
                    st.warning(err_msg)
        
        elif current_n == 3:
            st.header("Problem Visualization (3D)")
            # Call the new 3D plotting function
            # Ensure solution has length 3
            sol_3d = solution[:3] if len(solution) >= 3 else np.pad(solution, (0, 3-len(solution)))
            fig_3d = plot_lp_problem_3d(viz_c, viz_A, viz_b, sol_3d)
            if fig_3d: # Check if plot was created
                st.plotly_chart(fig_3d, use_container_width=True)
        
        else:
            st.info(f"Visualization is only available for 2 or 3 variables (Problem has {current_n}).")

        # Add sensitivity analysis section if the problem is solved
        if st.session_state.solver is not None and st.session_state.original_params is not None:
            # Display sensitivity analysis results
            display_sensitivity_analysis(
                st.session_state.solver,
                st.session_state.original_params['c'],
                st.session_state.original_params['A'],
                st.session_state.original_params['b'],
                use_fractions
            )

    # Clear resampled parameters if we didn't solve (this could happen if the user modified the form)
    if not should_solve and st.session_state.sensitivity_resample is not None:
        st.session_state.sensitivity_resample = None

if __name__ == "__main__":
    main()
