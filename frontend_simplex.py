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


def format_lp_problem(c, A, b, n, m):
    latex = r"\begin{align*}"

    # Objective function
    obj_terms = ' '.join([f'{c[i]:+}x_{i + 1}' if c[i] >= 0 else f'{c[i]}x_{i + 1}' for i in range(n)])
    latex += f"\\min \\quad & z = {obj_terms} \\\\[1em]"  # Add vertical space here

    # Add s.t. aligned under min
    latex += r"\text{s.t.} \quad & "

    # Add the first constraint
    constraint = f"{' + '.join([f'{A[0, j]}x_{j + 1}' for j in range(n)])} \\leq {b[0]}"
    latex += f"\qquad {constraint} \\\\"

    # Add the remaining constraints, each aligned with the s.t.
    for i in range(1, m):
        constraint = f"{' + '.join([f'{A[i, j]}x_{j + 1}' for j in range(n)])} \\leq {b[i]}"
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
    Create a visualization of a 2D linear programming problem with proper feasible region.

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

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure containing the visualization
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.path import Path
    import streamlit as st

    if len(c) != 2:
        st.error("Visualization is only available for problems with 2 variables.")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine appropriate plot bounds
    if solution is not None:
        max_x = max(12, solution[0] * 1.5)
        max_y = max(12, solution[1] * 1.5)
    else:
        max_x = max_y = 12

    # Extend bounds based on constraints
    for i in range(len(b)):
        if b[i] <= 0:
            continue
        if A[i, 0] > 0:
            max_x = max(max_x, b[i] / A[i, 0] * 1.5)
        if A[i, 1] > 0:
            max_y = max(max_y, b[i] / A[i, 1] * 1.5)

    # Ensure reasonable bounds
    max_x = min(max_x, 30)
    max_y = min(max_y, 30)

    # Calculate vertices of the feasible region
    vertices = []

    # Add the origin
    vertices.append((0, 0))

    # Add intersection of each constraint with x-axis (if positive)
    for i in range(len(b)):
        if A[i, 0] > 0 and b[i] > 0:
            x_intercept = b[i] / A[i, 0]
            vertices.append((x_intercept, 0))

    # Add intersection of each constraint with y-axis (if positive)
    for i in range(len(b)):
        if A[i, 1] > 0 and b[i] > 0:
            y_intercept = b[i] / A[i, 1]
            vertices.append((0, y_intercept))

    # Add intersections between pairs of constraints
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            # Skip if either constraint has zero coefficients
            if (A[i, 0] == 0 and A[i, 1] == 0) or (A[j, 0] == 0 and A[j, 1] == 0):
                continue

            # Solve the system of equations to find intersection
            # a1*x + b1*y = c1
            # a2*x + b2*y = c2
            a1, b1 = A[i]
            a2, b2 = A[j]
            c1, c2 = b[i], b[j]

            # Check if lines are parallel
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-10:
                continue

            # Calculate intersection point
            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det

            # Only add points in the first quadrant
            if x >= 0 and y >= 0:
                vertices.append((x, y))

    # Ensure we have unique vertices
    vertices = list(set(vertices))

    # Check each vertex against all constraints
    feasible_vertices = []
    for v in vertices:
        x, y = v
        feasible = True

        for i in range(len(b)):
            if A[i, 0] * x + A[i, 1] * y > b[i] + 1e-10:  # Add small tolerance
                feasible = False
                break

        if feasible:
            feasible_vertices.append(v)

    # Sort vertices to form a convex hull (Graham scan algorithm)
    if len(feasible_vertices) >= 3:
        # Find the lowest point
        lowest = min(feasible_vertices, key=lambda v: (v[1], v[0]))

        # Sort by polar angle
        def polar_angle(p):
            return np.arctan2(p[1] - lowest[1], p[0] - lowest[0])

        sorted_vertices = sorted(feasible_vertices, key=polar_angle)

        # Create a polygon for the feasible region
        polygon = patches.Polygon(sorted_vertices, closed=True,
                                  facecolor='gray', alpha=0.4, edgecolor=None)
        ax.add_patch(polygon)
    elif len(feasible_vertices) > 0:
        # For 1 or 2 vertices, just plot points
        x_vals = [v[0] for v in feasible_vertices]
        y_vals = [v[1] for v in feasible_vertices]
        ax.scatter(x_vals, y_vals, color='gray', alpha=0.5)

    # Draw non-negativity constraints
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2.5, label='x₁ ≥ 0')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2.5, label='x₂ ≥ 0')

    # Draw each constraint
    constraint_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for i in range(len(b)):
        a1, a2 = A[i]
        if a1 == 0 and a2 == 0:
            continue  # Skip degenerate constraints

        color = constraint_colors[i % len(constraint_colors)]

        if a2 == 0:  # Vertical line
            x_val = b[i] / a1
            if 0 <= x_val <= max_x:
                ax.axvline(x=x_val, color=color, linewidth=2.5,
                           label=f'Constraint {i + 1}: {a1}x₁ {"+" if a2 >= 0 else "-"} {abs(a2)}x₂ ≤ {b[i]}')
        elif a1 == 0:  # Horizontal line
            y_val = b[i] / a2
            if 0 <= y_val <= max_y:
                ax.axhline(y=y_val, color=color, linewidth=2.5,
                           label=f'Constraint {i + 1}: {a1}x₁ {"+" if a2 >= 0 else "-"} {abs(a2)}x₂ ≤ {b[i]}')
        else:  # Regular line
            # Generate x and y values for the line
            x_line = np.linspace(0, max_x, 1000)
            y_line = (b[i] - a1 * x_line) / a2

            # Only plot the line within the bounds of the plot
            mask = (y_line >= 0) & (y_line <= max_y)
            if np.any(mask):
                ax.plot(x_line[mask], y_line[mask], color=color, linewidth=2.5,
                        label=f'Constraint {i + 1}: {a1}x₁ {"+" if a2 >= 0 else "-"} {abs(a2)}x₂ ≤ {b[i]}')

    # Create a fine grid for objective function contours
    x = np.linspace(0, max_x, 100)
    y = np.linspace(0, max_y, 100)
    X, Y = np.meshgrid(x, y)

    # Compute objective function values
    Z = c[0] * X + c[1] * Y

    # For visualization, we want contours to increase toward the optimum
    if c[0] < 0 or c[1] < 0:  # Minimization with negative coefficients
        Z_contour = -Z  # Negate for contours only
    else:
        Z_contour = Z

    # Draw contours
    levels = np.linspace(np.min(Z_contour), np.max(Z_contour), 8)
    contours = ax.contour(X, Y, Z_contour, levels=levels, alpha=0.7, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=9, fmt='%.1f')

    # Mark the optimal solution if provided
    if solution is not None:
        ax.scatter(solution[0], solution[1], color='red', s=150, marker='*',
                   label=f'Optimal Solution ({solution[0]:.2f}, {solution[1]:.2f})')

        # Calculate the true objective value
        true_obj_value = c[0] * solution[0] + c[1] * solution[1]

        # Annotate with the correct objective value
        ax.annotate(f'Objective value: {true_obj_value:.2f}',
                    xy=(solution[0], solution[1]),
                    xytext=(solution[0] + max_x * 0.05, solution[1] + max_y * 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    # Set labels and title
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

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
    st.latex(format_lp_problem(c, A, b, n, m))

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

            # Create and display the visualization
            fig = plot_lp_problem(c, A, b, solution, "LP Problem Visualization")

            # Display the figure
            if fig is not None:
                st.pyplot(fig)

if __name__ == "__main__":
    main()