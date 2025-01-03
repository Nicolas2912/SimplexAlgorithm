import streamlit as st
import numpy as np
from fractions import Fraction
from tabulate import tabulate
from simplex import PrimalSimplex  # Assuming the class is in primal_simplex.py
import pandas as pd

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


def display_iteration(iteration, tableau, solver, use_fractions):
    """Display the tableau for a given iteration"""
    st.write(f"**Iteration {iteration}**")

    # Create DataFrame for display
    headers = [""] + [f"x{i + 1}" for i in range(solver.n)] + \
              [f"s{i + 1}" for i in range(solver.m)] + ["RHS"]

    # Format tableau data
    if use_fractions:
        formatted_data = np.vectorize(lambda x: str(solver._limit_fraction(Fraction(x))))(tableau)
    else:
        formatted_data = np.vectorize(lambda x: f"{x:.4f}")(tableau)

    # Create rows with labels
    rows = []
    rows.append(["z"] + list(formatted_data[0, :]))
    for i in range(1, solver.m + 1):
        rows.append([f"R{i}"] + list(formatted_data[i, :]))

    df = pd.DataFrame(rows, columns=headers)

    # Style the DataFrame
    styled_df = df.style.set_properties(**{
        'text-align': 'right',
        'font-family': 'monospace',
        'padding': '5px 10px',
        'color': '#000000',  # Black text color
        'background-color': '#f8f9fa',  # Light background
        'border': '1px solid #dee2e6'
    }).set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('background-color', '#e9ecef'),  # Slightly darker header background
                ('color', '#000000'),  # Black header text
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '5px 10px'),
                ('border', '1px solid #dee2e6')
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('border', '1px solid #dee2e6')
            ]
        }
    ])

    # Add zebra-striping for better readability
    styled_df = styled_df.apply(lambda _: ['background-color: #ffffff' if i % 2 == 0
                                           else 'background-color: #f8f9fa' for i in range(len(df))],
                                axis=0)

    st.write(styled_df)

    # Create LaTeX format for basis variables
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

def main():
    st.title("Linear Programming Solver")
    st.write("Solve linear programming problems using the Primal Simplex method")

    # Sidebar for problem setup
    st.sidebar.header("Problem Setup")

    # Option to use example problem
    use_example = st.sidebar.checkbox("Use example problem")

    if use_example:
        c, A, b = create_example_problem()
        m, n = A.shape
    else:
        # Get dimensions
        col1, col2 = st.sidebar.columns(2)
        m = col1.number_input("Number of constraints (m)", min_value=1, value=3)
        n = col2.number_input("Number of variables (n)", min_value=1, value=2)

        # Input for objective function coefficients
        st.sidebar.subheader("Objective Function Coefficients (c)")
        c_input = st.sidebar.text_input("Enter c values (comma-separated)", "2,3")
        try:
            c = np.array([float(x.strip()) for x in c_input.split(",")])
        except:
            st.error("Invalid input for objective function coefficients")
            return

        # Input for constraint matrix A
        st.sidebar.subheader("Constraint Matrix (A)")
        A = np.zeros((m, n))
        A_input = st.sidebar.text_area(
            "Enter A matrix (one row per line, comma-separated)",
            "1,2\n2,1\n1,1"
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
        b_input = st.sidebar.text_input("Enter b values (comma-separated)", "10,8,5")
        try:
            b = np.array([float(x.strip()) for x in b_input.split(",")])
        except:
            st.error("Invalid input for RHS values")
            return

    # Fraction display options
    use_fractions = st.sidebar.checkbox("Use fractions", value=True)
    fraction_digits = st.sidebar.number_input(
        "Maximum fraction digits",
        min_value=1,
        max_value=5,
        value=3,
        disabled=not use_fractions
    )

    # Validate inputs
    valid, message = validate_inputs(c, A, b)
    if not valid:
        st.error(message)
        return

    # Display the problem
    st.header("Problem Formulation")
    st.write("Minimize:")
    st.latex(f"z = {' + '.join([f'{c[i]}x_{i + 1}' for i in range(n)])}")

    st.write("Subject to:")
    for i in range(m):
        st.latex(f"{' + '.join([f'{A[i, j]}x_{j + 1}' for j in range(n)])} \\leq {b[i]}")
    st.latex("x_i \\geq 0 \\quad \\forall i")

    # Solve button
    if st.button("Solve"):
        st.header("Solution")
        try:
            # Create solver instance
            solver = PrimalSimplex(c, A, b, use_fractions=use_fractions, fraction_digits=fraction_digits)

            # Initialize tableau display
            st.markdown("### Solution Progress")
            iteration = 0
            display_iteration(iteration, solver.tableau, solver, use_fractions)

            # Solve step by step
            while True:
                pivot_col = solver._find_pivot_column()
                if pivot_col is None:
                    st.success("Optimal solution found!")
                    break

                pivot_row = solver._find_pivot_row(pivot_col)
                if pivot_row is None:
                    st.error("Problem is unbounded")
                    return

                # Show pivot information with latex variables
                st.write(f"Pivot: Entering Variable = $x_{pivot_col + 1}$, Leaving Row = $R_{pivot_row}$")

                # Perform pivot
                solver._pivot(pivot_row, pivot_col)
                iteration += 1
                display_iteration(iteration, solver.tableau, solver, use_fractions)

            # Extract solution
            solution = np.zeros(solver.n)
            for i in range(solver.n):
                col = solver.tableau[:, i]
                if np.sum(col == 1) == 1 and np.sum(col == 0) == solver.m:
                    row_idx = np.where(col == 1)[0][0]
                    if row_idx < len(solver.tableau):
                        solution[i] = solver.tableau[row_idx, -1]

            optimal_value = -solver.tableau[0, -1]

            # Display solution in LaTeX format
            st.markdown("### Optimal Solution")

            # Create the solution vector in LaTeX format
            solution_latex = "\\begin{align*}\n"
            solution_latex += "\\mathbf{x}^* &= \\begin{pmatrix}\n"
            solution_latex += " \\\\ ".join([f"{val:.4f}" for val in solution])
            solution_latex += "\n\\end{pmatrix} \\\\\n"
            solution_latex += f"z^* &= {optimal_value:.4f}\n"
            solution_latex += "\\end{align*}"

            st.latex(solution_latex)

        except Exception as e:
            st.error(f"Error solving problem: {str(e)}")
            st.write("Debug information:")
            st.write(f"Tableau shape: {solver.tableau.shape if hasattr(solver, 'tableau') else 'N/A'}")
            st.write(f"Number of variables (n): {solver.n if hasattr(solver, 'n') else 'N/A'}")
            st.write(f"Number of constraints (m): {solver.m if hasattr(solver, 'm') else 'N/A'}")

if __name__ == "__main__":
    main()