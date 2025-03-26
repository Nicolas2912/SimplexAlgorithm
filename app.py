# app.py
import streamlit as st
import numpy as np
import time # For sidebar hack delay

# Import from our modules
from simplex import PrimalSimplex, SensitivityAnalysis
from utils import (
    SolutionStorage,
    # create_example_problem, # Remove old import if exists
    create_example_2d,      # Import new functions
    create_example_3d,      # Import new functions
    validate_inputs,
    initialize_session_state
)
from ui_components import (
    format_lp_problem, display_iteration, display_solution
)
from plotting import (
    plot_lp_problem_2d, plot_lp_problem_3d, get_plot_bounds
)
from sensitivity_ui import display_sensitivity_analysis


# --- Function to load example data into session state ---
def load_example_into_state(c_ex, A_ex, b_ex):
    """Updates session state with example c, A, b and dimensions."""
    m_ex, n_ex = A_ex.shape
    st.session_state.m = m_ex
    st.session_state.n = n_ex
    # Use .astype(float) for consistent string conversion
    st.session_state.c_input = ",".join(map(str, c_ex.astype(float)))
    st.session_state.A_input = "\n".join(",".join(map(str, row.astype(float))) for row in A_ex)
    st.session_state.b_input = ",".join(map(str, b_ex.astype(float)))
    # Reset relevant states when loading an example
    st.session_state.eq_constraints = False # Assuming examples use <=
    st.session_state.has_solved = False
    st.session_state.solution_storage = None
    st.session_state.solver = None
    st.session_state.original_params = None
    st.session_state.sensitivity_resample = None
    st.session_state.auto_solve = False
    st.session_state.tableaus = []
    st.toast("Example loaded!") # Give user feedback


# --- Main Application Logic ---
def main():
    st.set_page_config(layout="wide") # Use wide layout
    initialize_session_state() # Ensure all keys exist

    st.title("Interactive Linear Programming Solver")
    st.write("Solve LP problems using the Primal Simplex method and explore sensitivity.")

    # --- Sidebar for Input ---
    with st.sidebar:
        st.header("Problem Definition")

        # --- Example Buttons ---
        st.write("Load Standard Examples:")
        col_ex1, col_ex2 = st.columns(2)
        if col_ex1.button("Load 2D Example", key="load_2d_example", use_container_width=True):
            c_ex, A_ex, b_ex = create_example_2d()
            load_example_into_state(c_ex, A_ex, b_ex)
            # No need to manually rerun, button click does it

        if col_ex2.button("Load 3D Example", key="load_3d_example", use_container_width=True):
            c_ex, A_ex, b_ex = create_example_3d()
            load_example_into_state(c_ex, A_ex, b_ex)
            # No need to manually rerun, button click does it

        st.markdown("---") # Separator
        st.write("Or Define Custom Problem:")

        # --- Manual Input Fields ---
        # REMOVED 'disabled' argument - inputs should always be editable unless solving
        col1, col2 = st.columns(2)
        m = col1.number_input("Constraints (m)", min_value=1, value=st.session_state.m, key='m_input')
        n = col2.number_input("Variables (n)", min_value=1, value=st.session_state.n, key='n_input')
        # Update state from input widgets immediately (Streamlit handles this)
        st.session_state.m = m
        st.session_state.n = n

        # Use the keys previously defined, ensure they read from session state
        c_input_str = st.text_input("Objective Coefficients (c)", value=st.session_state.c_input, key='c_input_widget', help="Comma-separated values, e.g., -2,-3,-4")
        A_input_str = st.text_area("Constraint Matrix (A)", value=st.session_state.A_input, key='A_input_widget', height=100, help="One row per line, comma-separated values, e.g.,\n1,1,1\n2,1,0")
        b_input_str = st.text_input("RHS Values (b)", value=st.session_state.b_input, key='b_input_widget', help="Comma-separated values, e.g., 6,4,7")

        # Update state from text inputs if modified by user
        # This happens automatically if 'value' is linked to session state
        # but we can explicitly update to be sure (though slightly redundant)
        st.session_state.c_input = c_input_str
        st.session_state.A_input = A_input_str
        st.session_state.b_input = b_input_str


        # --- Solver Options ---
        st.header("Solver Options")
        st.session_state.use_fractions = st.checkbox("Use fractions", value=st.session_state.use_fractions, key='use_fractions_widget')
        st.session_state.fraction_digits = st.number_input(
            "Max fraction digits", min_value=1, max_value=7,
            value=st.session_state.fraction_digits,
            disabled=not st.session_state.use_fractions,
            key='fraction_digits_widget'
        )
        st.session_state.eq_constraints = st.checkbox("Use equality constraints (=)", value=st.session_state.eq_constraints, key='eq_constraints_widget', help="Check if all constraints are equalities.")
    # --- Input Parsing and Validation ---
    try:
        # Use the current state values for parsing
        c = np.array([float(x.strip()) for x in st.session_state.c_input.split(',') if x.strip()], dtype=float)
        A = np.array([[float(x.strip()) for x in row.split(',') if x.strip()]
                      for row in st.session_state.A_input.strip().split('\n') if row.strip()], dtype=float)
        b = np.array([float(x.strip()) for x in st.session_state.b_input.split(',') if x.strip()], dtype=float)

        # Validate shapes against m, n from state
        valid, message = validate_inputs(c, A, b, st.session_state.m, st.session_state.n)
        if not valid:
            st.error(f"Input Error: {message}")
            st.stop() # Halt execution if inputs invalid
        st.session_state.problem_params = (c, A, b) # Store valid params

    except ValueError:
        st.error("Invalid numerical input. Ensure coefficients and RHS are comma-separated numbers.")
        st.stop()
    except Exception as e:
        st.error(f"Error parsing inputs: {e}")
        st.stop()


    # --- Display Problem ---
    st.header("Problem Formulation")
    try:
        st.latex(format_lp_problem(c, A, b, n, m, st.session_state.eq_constraints))
    except Exception as e:
        st.warning(f"Could not display problem formulation in LaTeX: {e}")


    # --- Solve ---
    # Determine if solving should happen (button press or auto-solve flag)
    solve_pressed = st.button("Solve Problem", key='solve_button')
    should_solve = solve_pressed or st.session_state.auto_solve

    if should_solve:
        st.session_state.auto_solve = False # Reset flag
        st.session_state.has_solved = False # Reset solve status
        st.session_state.tableaus = []      # Clear previous steps
        st.session_state.solution_storage = None
        st.session_state.solver = None
        st.session_state.original_params = None # Clear previous original params

        st.info("Solving...")
        solve_placeholder = st.empty()

        try:
            # Use copies of the validated parameters
            c_solve, A_solve, b_solve = [p.copy() for p in st.session_state.problem_params]

            # Create solver instance
            solver = PrimalSimplex(
                c_solve, A_solve, b_solve,
                use_fractions=st.session_state.use_fractions,
                fraction_digits=st.session_state.fraction_digits,
                eq_constraints=st.session_state.eq_constraints
            )

            # Store original parameters used for this solve run
            st.session_state.original_params = {'c': c_solve, 'A': A_solve, 'b': b_solve}

            # --- Simplex Iterations ---
            iteration = 0
            # Store initial state (Iteration 0)
            st.session_state.tableaus.append((iteration, solver.tableau.copy(), solver, None)) # Pass solver ref

            while True:
                pivot_col = solver._find_pivot_column()
                if pivot_col is None: # Optimal or infeasible (check tableau)
                    break

                pivot_row = solver._find_pivot_row(pivot_col)
                if pivot_row is None: # Unbounded
                    st.error("Problem is unbounded.")
                    # Store the last tableau before unboundedness detected
                    st.session_state.tableaus.append((iteration + 1, solver.tableau.copy(), solver, (pivot_col, None)))
                    st.session_state.solver = solver # Keep solver state for inspection
                    st.stop() # Stop execution for unbounded

                # Store pivot info before pivoting
                pivot_info = (pivot_col, pivot_row)

                # Perform pivot
                solver._pivot(pivot_row, pivot_col)
                iteration += 1

                # Store state after pivot
                st.session_state.tableaus.append((iteration, solver.tableau.copy(), solver, pivot_info))

                # Optional: Add a check for max iterations
                if iteration > 100: # Safety break
                    st.warning("Solver exceeded maximum iterations (100).")
                    break

            # --- Process Final State ---
            # If the loop finished without stopping due to unboundedness,
            # we assume it reached an optimal state (or potentially infeasible if Phase I was needed but not fully implemented/checked).
            # The unbounded case is handled *inside* the loop.
            final_tableau = solver.tableau
            is_optimal = True # Assume optimal if loop finished normally

            # Add a check here if your PrimalSimplex handles Phase I and artificial variables
            # Example (you might need to adapt this based on your simplex.py):
            # if hasattr(solver, 'artificial_indices') and solver.artificial_indices:
            #    for r in range(1, final_tableau.shape[0]):
            #        basis_map = solver.get_basis_variable_indices() # Get current basis map
            #        if r in basis_map and basis_map[r] in solver.artificial_indices:
            #             # Check if artificial variable is still basic with positive value
            #             if not np.isclose(final_tableau[r, -1], 0):
            #                  is_optimal = False
            #                  solve_placeholder.error("Problem is infeasible (artificial variable in basis).")
            #                  st.session_state.has_solved = False
            #                  st.session_state.solver = solver
            #                  break # Exit the check

            if is_optimal:
                solve_placeholder.success("Optimal solution found!")
                st.session_state.has_solved = True
                st.session_state.solver = solver # Store the solved instance

                # Extract solution
                solution = np.zeros(solver.n)
                # Use the helper function assumed to be in simplex.py (or add it)
                try:
                    basis_map = solver.get_basis_variable_indices()
                    for row_idx, var_idx in basis_map.items():
                        if var_idx < solver.n: # If it's an original variable
                            # Convert potential Fraction to float for storage if needed
                            try:
                                solution[var_idx] = float(final_tableau[row_idx, -1])
                            except TypeError: # Handle if it's already a float/int
                                solution[var_idx] = final_tableau[row_idx, -1]
                except AttributeError:
                    st.warning("Solver object missing 'get_basis_variable_indices'. Cannot reliably extract solution.")
                    solution.fill(np.nan) # Mark solution as unknown
                except Exception as basis_err:
                    st.warning(f"Error extracting basis variables: {basis_err}")
                    solution.fill(np.nan)

                optimal_value = -float(final_tableau[0, -1]) # Ensure float

                # Store results
                st.session_state.solution_storage = SolutionStorage(solution, optimal_value)

            # Note: The 'Unbounded' case is handled inside the loop with st.stop()
            # Note: The 'Infeasible' case might need more robust checking depending on simplex.py impl.

            else: # Cycle or other issue
                 solve_placeholder.warning(f"Solver finished with status: {status}")
                 st.session_state.has_solved = False
                 st.session_state.solver = solver


            # Clear the re-solve parameters now that solving is done
            st.session_state.sensitivity_resample = None


        except Exception as e:
            import traceback
            solve_placeholder.error(f"Error during solving process: {e}")
            st.error(traceback.format_exc()) # Show full traceback for debugging
            # Reset state on error
            st.session_state.has_solved = False
            st.session_state.solution_storage = None
            st.session_state.solver = None
            st.session_state.original_params = None
            st.session_state.sensitivity_resample = None
            st.stop()


    # --- Display Results (if solved) ---
    if st.session_state.has_solved and st.session_state.solver:
        # --- Solution Display ---
        display_solution(
            st.session_state.solver,
            st.session_state.solution_storage,
            st.session_state.use_fractions,
            st.session_state.fraction_digits
        )

        # --- Iteration Progress (Optional) ---
        st.session_state.show_progress = st.checkbox("Show Simplex Iterations", value=st.session_state.show_progress, key='show_progress_widget')
        if st.session_state.show_progress and st.session_state.tableaus:
            with st.expander("Simplex Tableaus", expanded=False):
                for iter_num, tableau_data, solver_ref, pivot_info in st.session_state.tableaus:
                    display_iteration(
                        iter_num, tableau_data, solver_ref,
                        st.session_state.use_fractions,
                        st.session_state.fraction_digits,
                        pivot_info
                    )
                    st.markdown("---") # Separator

        # --- Visualization ---
        st.header("Visualization")
        current_n = st.session_state.solver.n
        solution_vec = st.session_state.solution_storage.decimal_solution
        # Use original params for visualization consistency
        viz_c, viz_A, viz_b = [st.session_state.original_params[k] for k in ['c', 'A', 'b']]
        path_vertices = getattr(st.session_state.solver, 'path_vertices', None) # If solver stores path


        if current_n == 2:
             plot_bounds_2d = get_plot_bounds(viz_A, viz_b, solution_vec, path_vertices)
             fig_2d, err_msg = plot_lp_problem_2d(
                 viz_c, viz_A, viz_b, solution_vec[:2],
                 path_vertices=path_vertices,
                 title="LP Feasible Region & Simplex Path (2D)",
                 min_x=plot_bounds_2d[0], max_x=plot_bounds_2d[1],
                 min_y=plot_bounds_2d[2], max_y=plot_bounds_2d[3]
             )
             if fig_2d:
                 st.pyplot(fig_2d)
             elif err_msg:
                 st.warning(f"Could not generate 2D plot: {err_msg}")

        elif current_n == 3:
             fig_3d = plot_lp_problem_3d(
                 viz_c, viz_A, viz_b, solution_vec[:3],
                 path_vertices=path_vertices,
                 title="LP Feasible Region & Simplex Path (3D)"
             )
             if fig_3d:
                 st.plotly_chart(fig_3d, use_container_width=True)
             else:
                  st.warning("Could not generate 3D plot.") # plot_lp_problem_3d handles internal warnings

        else:
            st.info(f"Visualization is available for 2 or 3 variables (problem has {current_n}).")


        # --- Sensitivity Analysis ---
        # Check if original params are available (should be if solved successfully)
        if st.session_state.original_params:
             display_sensitivity_analysis(
                 st.session_state.solver,
                 st.session_state.original_params['c'],
                 st.session_state.original_params['A'],
                 st.session_state.original_params['b'],
                 st.session_state.use_fractions,
                 st.session_state.fraction_digits
             )
        else:
            st.warning("Original parameters not stored, cannot perform sensitivity analysis.")

    elif st.session_state.solver and not st.session_state.has_solved:
         # If solver exists but didn't reach optimal (infeasible/unbounded/error)
         st.warning(f"Solver finished, but no optimal solution found (Status: {st.session_state.solver.get_status()}).")
         # Optionally show final tableau for debugging
         if st.checkbox("Show Final Tableau (Non-Optimal)", value=False):
             final_iter = len(st.session_state.tableaus) - 1
             if final_iter >= 0:
                  _, tableau_data, solver_ref, pivot_info = st.session_state.tableaus[-1]
                  st.write(f"**Final State (Iteration {final_iter})**")
                  display_iteration(
                       final_iter, tableau_data, solver_ref,
                       st.session_state.use_fractions,
                       st.session_state.fraction_digits,
                       pivot_info
                  )


if __name__ == "__main__":
    main()