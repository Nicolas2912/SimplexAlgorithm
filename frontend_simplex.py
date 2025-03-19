import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from tabulate import tabulate
from simplex import PrimalSimplex  # Assuming the class is in primal_simplex.py
import pandas as pd


class SolutionStorage:
    def __init__(self, decimal_solution, decimal_optimal):
        self.decimal_solution = decimal_solution
        self.decimal_optimal = decimal_optimal
        # self.fraction_solution = fraction_solution
        # self.fraction_optimal = fraction_optimal


def convert_to_fraction(value, solver, fraction_digits):
    """Convert a decimal value to a fraction string with given digit limit"""
    frac = solver._limit_fraction(Fraction(float(value)))
    # Get numerator and denominator
    num, den = frac.numerator, frac.denominator
    # Limit the size of both numerator and denominator based on fraction_digits
    max_value = 10 ** fraction_digits
    if abs(num) > max_value or den > max_value:
        return f"{float(value):.{fraction_digits}f}"
    return str(frac)


def create_example_problem():
    """Create a simple example LP problem"""
    c = np.array([-3, -2, -4, -1, -5])
    A = np.array([
        [2, 1, 3, 1, 2],  # Constraint 1
        [1, 2, 1, 3, 1],  # Constraint 2
        [3, 1, 2, 1, 4],  # Constraint 3
        [1, 1, 1, 1, 1],  # Constraint 4
    ])
    b = np.array([10, 8, 12, 6])
    return c, A, b


def format_lp_problem(c, A, b, n, m, eq_constraints=False):
    latex = r"\begin{align*}"

    # Objective function
    obj_terms = ' '.join([f'{c[i]:+}x_{i + 1}' if c[i] >= 0 else f'{c[i]}x_{i + 1}' for i in range(n)])
    latex += f"\\min \\quad & z = {obj_terms} \\\\[1em]"  # Add vertical space here

    # Add s.t. aligned under min
    latex += r"\text{s.t.} \quad & "

    # Choose equality or inequality symbol based on eq_constraints flag
    constraint_symbol = "=" if eq_constraints else "\\leq"

    # Add the first constraint
    constraint = f"{' + '.join([f'{A[0, j]}x_{j + 1}' for j in range(n)])} {constraint_symbol} {b[0]}"
    latex += f"\qquad {constraint} \\\\"

    # Add the remaining constraints, each aligned with the s.t.
    for i in range(1, m):
        constraint = f"{' + '.join([f'{A[i, j]}x_{j + 1}' for j in range(n)])} {constraint_symbol} {b[i]}"
        latex += f"& \qquad {constraint} \\\\"

    # Add non-negativity constraint
    latex += r"& \qquad x_i \geq 0 \quad \forall i"

    # Close environment
    latex += r"\end{align*}"

    return latex


def display_matrix(matrix, name):
    """Display a matrix/vector in a more readable format"""
    st.write(f"{name}:")
    st.write(matrix)


def validate_inputs(c, A, b):
    """Validate input dimensions and values"""
    try:
        if len(c) != A.shape[1]:
            return False, "Number of variables in objective function doesn't match constraints"
        if len(b) != A.shape[0]:
            return False, "Number of constraints doesn't match RHS"
        return True, "Valid inputs"
    except:
        return False, "Invalid input format"


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


def plot_lp_problem(c, A, b, solution=None, title="Linear Programming Visualization"):
    """
    Create a robust visualization of a 2D linear programming problem.

    Parameters:
    -----------
    c : array-like
        Coefficients of the objective function (length 2)
    A : array-like
        Constraint coefficients matrix
    b : array-like
        Right-hand side of constraints
    solution : array-like, optional
        The optimal solution to highlight (length 2)
    title : str, optional
        Title for the plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Ensure we're working with numpy arrays
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if len(c) != 2:
        return None, "Visualization is only available for problems with 2 variables."

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate initial bounds - start with a reasonable default
    x_min, y_min = 0, 0
    x_max, y_max = 10, 10

    # Collect all potential vertices of the feasible region
    vertices = []

    # Always include origin if it's feasible
    origin_feasible = True
    for i in range(len(b)):
        if A[i, 0] * 0 + A[i, 1] * 0 > b[i] + 1e-10:
            origin_feasible = False
            break

    if origin_feasible:
        vertices.append((0, 0))

    # Find intersections with axes
    for i in range(len(b)):
        # x-axis intersection (y=0)
        if abs(A[i, 0]) > 1e-10:  # Avoid division by zero
            x_intercept = b[i] / A[i, 0]
            if x_intercept >= 0:
                y_val = 0
                # Check if point satisfies all constraints
                feasible = True
                for j in range(len(b)):
                    if A[j, 0] * x_intercept + A[j, 1] * y_val > b[j] + 1e-10:
                        feasible = False
                        break
                if feasible:
                    vertices.append((x_intercept, 0))
                    x_max = max(x_max, x_intercept * 1.2)

        # y-axis intersection (x=0)
        if abs(A[i, 1]) > 1e-10:  # Avoid division by zero
            y_intercept = b[i] / A[i, 1]
            if y_intercept >= 0:
                x_val = 0
                # Check if point satisfies all constraints
                feasible = True
                for j in range(len(b)):
                    if A[j, 0] * x_val + A[j, 1] * y_intercept > b[j] + 1e-10:
                        feasible = False
                        break
                if feasible:
                    vertices.append((0, y_intercept))
                    y_max = max(y_max, y_intercept * 1.2)

    # Find intersections between constraints
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            # Get coefficients
            a1, b1 = A[i, 0], A[i, 1]
            a2, b2 = A[j, 0], A[j, 1]
            c1, c2 = b[i], b[j]

            # Check if constraints are parallel
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-10:
                continue

            # Calculate intersection point
            try:
                x = (c1 * b2 - c2 * b1) / det
                y = (a1 * c2 - a2 * c1) / det

                # Check if point is in first quadrant
                if x >= 0 and y >= 0:
                    # Check if point satisfies all constraints
                    feasible = True
                    for k in range(len(b)):
                        if A[k, 0] * x + A[k, 1] * y > b[k] + 1e-10:
                            feasible = False
                            break

                    if feasible:
                        vertices.append((x, y))
                        x_max = max(x_max, x * 1.2)
                        y_max = max(y_max, y * 1.2)
            except:
                # Skip any numerical issues
                continue

    # If solution is provided, add it to the vertices and update bounds
    if solution is not None:
        x_sol, y_sol = solution[0], solution[1]
        vertices.append((x_sol, y_sol))
        x_max = max(x_max, x_sol * 1.2)
        y_max = max(y_max, y_sol * 1.2)

    # Ensure we have reasonable bounds
    x_max = max(10, x_max)
    y_max = max(10, y_max)

    # Draw non-negativity constraints
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, label='x₁ ≥ 0')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, label='x₂ ≥ 0')

    # Draw constraint lines
    constraint_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for i in range(len(b)):
        a1, a2 = A[i, 0], A[i, 1]
        if abs(a1) < 1e-10 and abs(a2) < 1e-10:
            continue  # Skip degenerate constraints

        color = constraint_colors[i % len(constraint_colors)]
        constraint_text = f"Constraint {i + 1}: "

        if abs(a1) > 1e-10:
            constraint_text += f"{a1}x₁"

        if abs(a2) > 1e-10:
            if a2 > 0 and abs(a1) > 1e-10:
                constraint_text += f" + {a2}x₂"
            else:
                constraint_text += f" {a2}x₂"

        constraint_text += f" ≤ {b[i]}"

        # Calculate points for the line
        if abs(a1) < 1e-10:  # Horizontal line
            y_val = b[i] / a2
            ax.axhline(y=y_val, color=color, linestyle='-', linewidth=1.5, label=constraint_text)
        elif abs(a2) < 1e-10:  # Vertical line
            x_val = b[i] / a1
            ax.axvline(x=x_val, color=color, linestyle='-', linewidth=1.5, label=constraint_text)
        else:
            # General line: calculate two points and draw
            if a2 > 0:  # Line crosses y-axis
                y_intercept = b[i] / a2
                points = [(0, y_intercept)]
            else:
                points = []

            if a1 > 0:  # Line crosses x-axis
                x_intercept = b[i] / a1
                points.append((x_intercept, 0))

            # If we need more points for very skewed lines
            if len(points) < 2:
                # Calculate additional point using x_max
                if abs(a2) > 1e-10:  # Avoid division by zero
                    y_at_xmax = (b[i] - a1 * x_max) / a2
                    if y_at_xmax >= 0:
                        points.append((x_max, y_at_xmax))

                # Calculate additional point using y_max
                if abs(a1) > 1e-10:  # Avoid division by zero
                    x_at_ymax = (b[i] - a2 * y_max) / a1
                    if x_at_ymax >= 0:
                        points.append((x_at_ymax, y_max))

            # Draw line if we have at least 2 points
            if len(points) >= 2:
                sorted_points = sorted(points, key=lambda p: p[0])  # Sort by x-coordinate
                xs, ys = zip(*sorted_points)
                ax.plot(xs, ys, color=color, linestyle='-', linewidth=1.5, label=constraint_text)

    # Create the feasible region polygon if we have vertices
    if len(vertices) >= 3:
        # Sort vertices to form a convex hull
        def polar_angle(p):
            return np.arctan2(p[1] - centroid_y, p[0] - centroid_x)

        centroid_x = sum(v[0] for v in vertices) / len(vertices)
        centroid_y = sum(v[1] for v in vertices) / len(vertices)

        sorted_vertices = sorted(vertices, key=polar_angle)

        # Create polygon
        polygon = patches.Polygon(sorted_vertices, closed=True,
                                  facecolor='lightgray', alpha=0.5, edgecolor='gray')
        ax.add_patch(polygon)

    # Mark optimal solution if provided
    if solution is not None:
        x_sol, y_sol = solution
        obj_value = c[0] * x_sol + c[1] * y_sol

        ax.scatter(x_sol, y_sol, color='red', s=100, marker='*',
                   label=f'Optimal Solution ({x_sol:.2f}, {y_sol:.2f})')

        ax.annotate(f'Objective value: {obj_value:.2f}',
                    xy=(x_sol, y_sol),
                    xytext=(x_sol + x_max * 0.05, y_sol + y_max * 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    # Draw objective function contours
    x_grid = np.linspace(0, x_max, 100)
    y_grid = np.linspace(0, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = c[0] * X + c[1] * Y

    # For minimization problems with negative coefficients, negate Z for visualization
    if c[0] < 0 or c[1] < 0:
        Z = -Z

    # Generate contour levels
    min_z = np.min(Z)
    max_z = np.max(Z)
    levels = np.linspace(min_z, max_z, 6)

    # Draw contours
    contours = ax.contour(X, Y, Z, levels=levels, alpha=0.7, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)

    # Set axis limits with a bit of padding
    padding = 0.05
    x_range = max(1, x_max - x_min)
    y_range = max(1, y_max - y_min)

    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

    # Set labels and title
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add legend with reasonable size
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    return fig


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


def main():
    # Initialize session state variables if they don't exist
    if 'solution_storage' not in st.session_state:
        st.session_state.solution_storage = None
    if 'tableaus' not in st.session_state:
        st.session_state.tableaus = []  # Will store tuples of (iteration, tableau, solver, pivot_info)
    if 'has_solved' not in st.session_state:
        st.session_state.has_solved = False

    st.title("Linear Programming Solver")
    st.write("Solve linear programming problems using the Primal Simplex method")

    # Sidebar for problem setup
    st.sidebar.header("Problem Setup")

    # Option to use example problem
    use_example = st.sidebar.checkbox("Use example problem", key='use_example')

    if use_example:
        c, A, b = create_example_problem()
        m, n = A.shape
    else:
        # Get dimensions
        col1, col2 = st.sidebar.columns(2)
        m = col1.number_input("Number of constraints (m)", min_value=1, value=3, key='m')
        n = col2.number_input("Number of variables (n)", min_value=1, value=2, key='n')

        # Input for objective function coefficients
        st.sidebar.subheader("Objective Function Coefficients (c)")
        c_input = st.sidebar.text_input("Enter c values (comma-separated)", "2,3", key='c_input')
        try:
            c = np.array([float(x.strip()) for x in c_input.split(",")])
        except:
            st.error("Invalid input for objective function coefficients")
            return

        # Input for constraint matrix A
        st.sidebar.subheader("Constraint Matrix (A)")
        A_input = st.sidebar.text_area(
            "Enter A matrix (one row per line, comma-separated)",
            "1,2\n2,1\n1,1",
            key='A_input'
        )
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
        b_input = st.sidebar.text_input("Enter b values (comma-separated)", "10,8,5", key='b_input')
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

    # Solve button
    if st.button("Solve", key='solve_button'):
        st.session_state.has_solved = True
        # Clear previous tableaus when solving new problem
        st.session_state.tableaus = []

        st.header("Solution Process")
        try:
            # Create solver instance with fractions
            solver = PrimalSimplex(c, A, b, use_fractions=True, fraction_digits=fraction_digits,
                                   eq_constraints=eq_constraints)

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
                if np.sum(col == 1) == 1 and np.sum(col == 0) == solver.m:
                    row_idx = np.where(col == 1)[0][0]
                    if row_idx < len(solver.tableau):
                        decimal_solution[i] = solver.tableau[row_idx, -1]

            decimal_optimal = -solver.tableau[0, -1]

            # Store only decimal values in solution storage
            st.session_state.solution_storage = SolutionStorage(
                decimal_solution,
                decimal_optimal
            )

        except Exception as e:
            st.error(f"Error solving problem: {str(e)}")
            return

    # Display solution progress if available
    if st.session_state.has_solved and st.session_state.tableaus:
        st.success("Optimal solution found!")

        # Option to show solution progress
        show_progress = st.checkbox("Show solution progress", value=True)

        if show_progress:
            st.markdown("### Solution Progress")
            for iteration, tableau, solver, pivot_info in st.session_state.tableaus:
                display_iteration(iteration, tableau, solver, use_fractions, fraction_digits, pivot_info)

    # Display final solution if available
    if st.session_state.solution_storage is not None:
        st.header("Optimal Solution")
        display_solution(solver, use_fractions)

        # Add visualization for 2D problems
        if len(c) == 2:
            st.header("Problem Visualization")

            # Get solution from storage
            solution = st.session_state.solution_storage.decimal_solution

            # Check if this is a large-scale problem
            is_large_scale = solution[0] > 100 or solution[1] > 100

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
                    # Create a matplotlib figure that shows the full problem
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as patches

                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Set bounds to fully contain the solution
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
                    st.pyplot(fig)

                else:  # Focus around solution
                    # Create a zoomed-in view focused on the solution area
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
                    st.pyplot(fig)
            else:
                # For normal-scale problems, use the standard visualization
                fig = plot_lp_problem(c, A, b, solution, "LP Problem Visualization")
                st.pyplot(fig)


if __name__ == "__main__":
    main()