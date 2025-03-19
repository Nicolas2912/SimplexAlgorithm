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
    Create a visualization of a 2D linear programming problem.

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
    if len(c) != 2:
        st.error("Visualization is only available for problems with 2 variables.")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set reasonable bounds for the plot
    x_min, x_max = -1, max(20, solution[0] * 1.5 if solution is not None else 20)
    y_min, y_max = -1, max(20, solution[1] * 1.5 if solution is not None else 20)

    # Extend bounds if needed to include important points
    if solution is not None:
        x_max = max(x_max, solution[0] * 1.2 + 1)
        y_max = max(y_max, solution[1] * 1.2 + 1)

    # Create a grid of points
    x = np.linspace(x_min, x_max, 1000)
    y = np.linspace(y_min, y_max, 1000)
    X, Y = np.meshgrid(x, y)

    # Plot and shade the feasible region
    region_mask = np.ones_like(X, dtype=bool)

    # Plot each constraint line
    for i in range(len(b)):
        if A[i, 0] == 0 and A[i, 1] == 0:
            continue  # Skip constraints with zero coefficients

        # Calculate the constraint line
        if A[i, 1] == 0:  # Vertical line
            constraint_x = b[i] / A[i, 0]
            ax.axvline(x=constraint_x, label=f'Constraint {i + 1}', linestyle='-')
            # Update mask
            if A[i, 0] > 0:
                region_mask &= (X <= constraint_x)
            else:
                region_mask &= (X >= constraint_x)
        else:  # Regular line
            constraint_y = (b[i] - A[i, 0] * x) / A[i, 1]
            ax.plot(x, constraint_y, label=f'Constraint {i + 1}')
            # Update mask
            if A[i, 1] > 0:
                region_mask &= (Y <= (b[i] - A[i, 0] * X) / A[i, 1])
            else:
                region_mask &= (Y >= (b[i] - A[i, 0] * X) / A[i, 1])

    # Add non-negativity constraints
    region_mask &= (X >= 0) & (Y >= 0)
    ax.axvline(x=0, color='black', linestyle='-', label='x₁ ≥ 0')
    ax.axhline(y=0, color='black', linestyle='-', label='x₂ ≥ 0')

    # Shade the feasible region
    ax.imshow(region_mask.T, extent=[x_min, x_max, y_min, y_max], origin='lower',
              alpha=0.3, cmap='Blues', aspect='auto')

    # Draw objective function contours
    Z = c[0] * X + c[1] * Y if c[0] >= 0 and c[1] >= 0 else -c[0] * X - c[1] * Y
    contour_levels = np.linspace(np.min(Z), np.max(Z), 10)
    contours = ax.contour(X, Y, Z, levels=contour_levels, alpha=0.6, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)

    # Mark the optimal solution if provided
    if solution is not None:
        ax.scatter(solution[0], solution[1], color='red', s=100, marker='*',
                   label=f'Optimal Solution ({solution[0]:.2f}, {solution[1]:.2f})')

        # Mark objective value at optimal solution
        obj_value = -c[0] * solution[0] - c[1] * solution[1] if c[0] < 0 or c[1] < 0 else c[0] * solution[0] + c[1] * \
                                                                                          solution[1]
        ax.annotate(f'Objective value: {obj_value:.2f}',
                    xy=(solution[0], solution[1]),
                    xytext=(solution[0] + 1, solution[1] + 1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    # Set labels and title
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

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

            # Add option to compare with SciPy solution
            from scipy.optimize import linprog

            col1, col2 = st.columns(2)
            with col1:
                show_scipy = st.checkbox("Compare with SciPy solution", value=True)

            fig = plot_lp_problem(c, A, b, solution, "LP Problem Visualization")

            # Add SciPy solution for comparison if requested
            if show_scipy and fig is not None:
                try:
                    # Solve with SciPy
                    result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

                    if result.success:
                        scipy_solution = result.x

                        # Add SciPy solution to the plot
                        ax = fig.gca()
                        ax.scatter(scipy_solution[0], scipy_solution[1], color='green', s=100, marker='o',
                                   label=f'SciPy Solution ({scipy_solution[0]:.2f}, {scipy_solution[1]:.2f})')

                        # Add annotation for SciPy solution
                        obj_value = -c[0] * scipy_solution[0] - c[1] * scipy_solution[1] if c[0] < 0 or c[1] < 0 else c[
                                                                                                                          0] * \
                                                                                                                      scipy_solution[
                                                                                                                          0] + \
                                                                                                                      c[
                                                                                                                          1] * \
                                                                                                                      scipy_solution[
                                                                                                                          1]
                        ax.annotate(f'SciPy obj: {obj_value:.2f}',
                                    xy=(scipy_solution[0], scipy_solution[1]),
                                    xytext=(scipy_solution[0] - 1, scipy_solution[1] - 1),
                                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5))

                        # Update legend
                        ax.legend(loc='upper right')

                        # Show comparison in text
                        st.write("**Solution Comparison:**")
                        st.write(
                            f"PrimalSimplex: ({solution[0]:.2f}, {solution[1]:.2f}) with objective value {st.session_state.solution_storage.decimal_optimal:.2f}")
                        st.write(
                            f"SciPy: ({scipy_solution[0]:.2f}, {scipy_solution[1]:.2f}) with objective value {obj_value:.2f}")
                except Exception as e:
                    st.error(f"Error comparing with SciPy: {str(e)}")

            # Display the figure
            if fig is not None:
                st.pyplot(fig)


if __name__ == "__main__":
    main()