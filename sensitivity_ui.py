# sensitivity_ui.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from simplex import SensitivityAnalysis # Assumes simplex.py is in the same directory or PYTHONPATH
from plotting import ( # Import plotting helpers
    plot_objective_sensitivity,
    plot_rhs_sensitivity,
    plot_combined_sensitivity,
    get_plot_bounds
)
from utils import convert_to_fraction # For displaying values


def display_sensitivity_analysis(solver, original_c, original_A, original_b, use_fractions, fraction_digits):
    """
    Display sensitivity analysis results in Streamlit tabs, including interactive elements.
    """
    if solver is None:
        st.warning("Solver instance not available for sensitivity analysis.")
        return

    try:
        sensitivity = SensitivityAnalysis(solver)
        optimal_solution = sensitivity.optimal_solution
        optimal_obj = sensitivity.optimal_obj
        is_2d = (solver.n == 2)
        plot_bounds = None
        if is_2d:
             # Calculate bounds once for all 2D plots in this section
             path_vertices = getattr(solver, 'path_vertices', None) # Get path if stored by solver
             plot_bounds = get_plot_bounds(original_A, original_b, optimal_solution, path_vertices)


    except Exception as e:
        st.error(f"Failed to perform sensitivity analysis: {e}")
        return

    st.header("Sensitivity Analysis")

    tab_titles = ["Objective Coefficients", "Right-Hand Side", "Shadow Prices", "What-If Analysis"]
    # Use st.session_state to remember the active tab
    active_tab_idx = st.session_state.get('active_tab_index', 0)

    # Create tabs. `st.tabs` returns a list of context managers.
    tabs = st.tabs(tab_titles)

    # --- Tab 1: Objective Coefficient Sensitivity ---
    with tabs[0]:
        st.subheader("Objective Function Coefficient Sensitivity")
        st.write("Allowable range for objective coefficients to maintain the current optimal basis.")
        obj_sensitivity = sensitivity.objective_sensitivity_analysis()
        obj_data = []
        for j in range(solver.n):
            current_value = original_c[j]
            if j in obj_sensitivity:
                lower, upper = obj_sensitivity[j]
                delta_lower_val = current_value - lower if lower != -np.inf else np.inf
                delta_upper_val = upper - current_value if upper != np.inf else np.inf

                obj_data.append({
                    "Variable": f"c{j + 1}",
                    "Current": convert_to_fraction(current_value, solver, fraction_digits, not use_fractions),
                    "Lower Bound": "-∞" if lower == -np.inf else convert_to_fraction(lower, solver, fraction_digits, not use_fractions),
                    "Upper Bound": "+∞" if upper == np.inf else convert_to_fraction(upper, solver, fraction_digits, not use_fractions),
                    "Allowable ↓": "Any" if np.isinf(delta_lower_val) else convert_to_fraction(delta_lower_val, solver, fraction_digits, not use_fractions),
                    "Allowable ↑": "Any" if np.isinf(delta_upper_val) else convert_to_fraction(delta_upper_val, solver, fraction_digits, not use_fractions)
                })
            else: # Non-basic variable at zero reduced cost (can happen) or issue
                 obj_data.append({
                    "Variable": f"c{j + 1}",
                    "Current": convert_to_fraction(current_value, solver, fraction_digits, not use_fractions),
                    "Lower Bound": "N/A", "Upper Bound": "N/A",
                    "Allowable ↓": "N/A", "Allowable ↑": "N/A"
                 })

        if obj_data:
            obj_df = pd.DataFrame(obj_data)
            st.dataframe(obj_df, hide_index=True, use_container_width=True)

        # Interactive Analysis (only if 2D)
        if is_2d and plot_bounds:
            st.markdown("---")
            st.subheader("Interactive Objective Coefficient Analysis (2D)")
            selected_var_obj = st.selectbox("Select variable to analyze:", [f"c{j + 1}" for j in range(solver.n)], key="obj_var_select")
            j_obj = int(selected_var_obj[1:]) - 1

            if j_obj in obj_sensitivity:
                current_val_obj = original_c[j_obj]
                lower_obj, upper_obj = obj_sensitivity[j_obj]
                # Define reasonable slider range around current value, respecting bounds
                slider_min_obj = max(lower_obj, current_val_obj - 5) if lower_obj != -np.inf else current_val_obj - 5
                slider_max_obj = min(upper_obj, current_val_obj + 5) if upper_obj != np.inf else current_val_obj + 5
                # Ensure min < max for slider
                if slider_max_obj <= slider_min_obj: slider_max_obj = slider_min_obj + 1

                new_val_obj = st.slider(f"Adjust {selected_var_obj}:", float(slider_min_obj), float(slider_max_obj), float(current_val_obj), step=0.1, key="obj_slider")

                is_within_range_obj = (lower_obj == -np.inf or new_val_obj >= lower_obj - 1e-6) and \
                                      (upper_obj == np.inf or new_val_obj <= upper_obj + 1e-6) # Add tolerance

                if is_within_range_obj:
                    delta_c = new_val_obj - current_val_obj
                    # Ensure optimal_solution is long enough
                    sol_j = optimal_solution[j_obj] if j_obj < len(optimal_solution) else 0
                    new_obj_val = optimal_obj + delta_c * sol_j

                    col1, col2 = st.columns(2)
                    col1.metric("Original Z*", f"{optimal_obj:.4f}")
                    col2.metric("New Z* (Predicted)", f"{new_obj_val:.4f}", delta=f"{new_obj_val - optimal_obj:.4f}")

                    new_c_vec = original_c.copy()
                    new_c_vec[j_obj] = new_val_obj
                    fig_obj = plot_objective_sensitivity(original_c, new_c_vec, optimal_solution, j_obj, optimal_obj, new_obj_val, plot_bounds)
                    if fig_obj: st.plotly_chart(fig_obj, use_container_width=True)

                else:
                    st.warning("Value is outside the allowable range. Basis may change, predicted Z* is not guaranteed.")
            else:
                st.info(f"Sensitivity range not available for {selected_var_obj}.")
        elif is_2d and not plot_bounds:
             st.warning("Could not determine plot bounds for interactive analysis.")
        elif not is_2d:
            st.info("Interactive visualization is available for 2D problems only.")


    # --- Tab 2: Right-Hand Side Sensitivity ---
    with tabs[1]:
        st.subheader("Right-Hand Side (RHS) Sensitivity")
        st.write("Allowable range for RHS values (b) to maintain the current optimal basis.")
        rhs_sensitivity = sensitivity.rhs_sensitivity_analysis()
        shadow_prices_rhs = sensitivity.shadow_prices()
        rhs_data = []
        for i in range(solver.m):
            current_value = original_b[i]
            shadow_price = shadow_prices_rhs[i] if i < len(shadow_prices_rhs) else 0

            if i in rhs_sensitivity:
                lower, upper = rhs_sensitivity[i]
                delta_lower_val = current_value - lower if lower != -np.inf else np.inf
                delta_upper_val = upper - current_value if upper != np.inf else np.inf

                rhs_data.append({
                    "Constraint": f"b{i + 1}",
                    "Current": convert_to_fraction(current_value, solver, fraction_digits, not use_fractions),
                    "Shadow Price": convert_to_fraction(shadow_price, solver, fraction_digits + 1, not use_fractions), # More digits for shadow price
                    "Lower Bound": "-∞" if lower == -np.inf else convert_to_fraction(lower, solver, fraction_digits, not use_fractions),
                    "Upper Bound": "+∞" if upper == np.inf else convert_to_fraction(upper, solver, fraction_digits, not use_fractions),
                    "Allowable ↓": "Any" if np.isinf(delta_lower_val) else convert_to_fraction(delta_lower_val, solver, fraction_digits, not use_fractions),
                    "Allowable ↑": "Any" if np.isinf(delta_upper_val) else convert_to_fraction(delta_upper_val, solver, fraction_digits, not use_fractions)
                })
            else: # Should generally have range if solved, but handle case
                 rhs_data.append({
                    "Constraint": f"b{i + 1}",
                    "Current": convert_to_fraction(current_value, solver, fraction_digits, not use_fractions),
                    "Shadow Price": convert_to_fraction(shadow_price, solver, fraction_digits + 1, not use_fractions),
                    "Lower Bound": "N/A", "Upper Bound": "N/A",
                    "Allowable ↓": "N/A", "Allowable ↑": "N/A"
                 })

        if rhs_data:
            rhs_df = pd.DataFrame(rhs_data)
            st.dataframe(rhs_df, hide_index=True, use_container_width=True)

        # Interactive Analysis (only if 2D)
        if is_2d and plot_bounds:
            st.markdown("---")
            st.subheader("Interactive RHS Analysis (2D)")
            selected_constr_rhs = st.selectbox("Select constraint RHS to analyze:", [f"b{i + 1}" for i in range(solver.m)], key="rhs_constr_select")
            i_rhs = int(selected_constr_rhs[1:]) - 1

            if i_rhs in rhs_sensitivity:
                current_val_rhs = original_b[i_rhs]
                lower_rhs, upper_rhs = rhs_sensitivity[i_rhs]
                shadow_p = shadow_prices_rhs[i_rhs] if i_rhs < len(shadow_prices_rhs) else 0

                # Define reasonable slider range around current value, respecting bounds
                slider_min_rhs = max(lower_rhs, current_val_rhs - 5) if lower_rhs != -np.inf else current_val_rhs - 5
                slider_max_rhs = min(upper_rhs, current_val_rhs + 5) if upper_rhs != np.inf else current_val_rhs + 5
                 # Ensure min < max for slider
                if slider_max_rhs <= slider_min_rhs: slider_max_rhs = slider_min_rhs + 1

                new_val_rhs = st.slider(f"Adjust {selected_constr_rhs}:", float(slider_min_rhs), float(slider_max_rhs), float(current_val_rhs), step=0.1, key="rhs_slider")

                is_within_range_rhs = (lower_rhs == -np.inf or new_val_rhs >= lower_rhs - 1e-6) and \
                                      (upper_rhs == np.inf or new_val_rhs <= upper_rhs + 1e-6) # Add tolerance

                if is_within_range_rhs:
                    delta_b = new_val_rhs - current_val_rhs
                    new_obj_val_rhs = optimal_obj + shadow_p * delta_b

                    col1, col2 = st.columns(2)
                    col1.metric("Original Z*", f"{optimal_obj:.4f}")
                    col2.metric("New Z* (Predicted)", f"{new_obj_val_rhs:.4f}", delta=f"{new_obj_val_rhs - optimal_obj:.4f}")

                    new_b_vec = original_b.copy()
                    new_b_vec[i_rhs] = new_val_rhs
                    fig_rhs = plot_rhs_sensitivity(original_A, original_b, new_b_vec, optimal_solution, i_rhs, plot_bounds)
                    if fig_rhs: st.plotly_chart(fig_rhs, use_container_width=True)

                else:
                    st.warning("Value is outside the allowable range. Basis may change, predicted Z* is not guaranteed.")
            else:
                 st.info(f"Sensitivity range not available for {selected_constr_rhs}.")
        elif is_2d and not plot_bounds:
             st.warning("Could not determine plot bounds for interactive analysis.")
        elif not is_2d:
            st.info("Interactive visualization is available for 2D problems only.")


    # --- Tab 3: Shadow Prices ---
    with tabs[2]:
        st.subheader("Shadow Prices (Dual Values)")
        st.write("""
        Shadow prices indicate the change in the optimal objective value per unit increase in the RHS of a constraint, *assuming the change is within the allowable range*.
        - Non-negative shadow price (for min problem): Increasing RHS makes objective worse (increase).
        - Non-positive shadow price (for min problem): Increasing RHS makes objective better (decrease).
        - Zero shadow price: Constraint is non-binding, small changes in RHS have no impact on Z*.
        """)
        shadow_prices_sp = sensitivity.shadow_prices()
        sp_data = []
        for i in range(solver.m):
            price = shadow_prices_sp[i] if i < len(shadow_prices_sp) else 0
            sp_data.append({
                "Constraint": f"b{i + 1}",
                "Shadow Price (πᵢ)": convert_to_fraction(price, solver, fraction_digits + 1, not use_fractions),
            })

        if sp_data:
            sp_df = pd.DataFrame(sp_data)
            st.dataframe(sp_df, hide_index=True, use_container_width=True)

            # Bar chart
            fig_sp = go.Figure(data=[
                go.Bar(x=[f"b{i + 1}" for i in range(solver.m)], y=shadow_prices_sp, marker_color='lightcoral')
            ])
            fig_sp.update_layout(
                title="Shadow Prices per Constraint",
                xaxis_title="Constraint RHS",
                yaxis_title="Shadow Price (π)",
                height=400
            )
            st.plotly_chart(fig_sp, use_container_width=True)


    # --- Tab 4: What-If Analysis ---
    with tabs[3]:
        st.subheader("What-If Analysis")
        st.write("""
        Explore the *predicted* impact of simultaneous changes to objective coefficients and RHS values,
        *provided all changes remain within their individual allowable ranges*. If any change goes outside its range,
        the basis may change, and a re-solve is needed for an accurate result.
        """)

        new_c_whatif = original_c.copy()
        new_b_whatif = original_b.copy()
        any_outside_range = False

        col1_wi, col2_wi = st.columns(2)

        # Objective Coefficient Sliders
        with col1_wi:
            st.markdown("**Objective Coefficients (c)**")
            obj_sens_whatif = sensitivity.objective_sensitivity_analysis() # Recalculate here if needed, but should be same
            for j in range(solver.n):
                current_c = original_c[j]
                lower_c, upper_c = obj_sens_whatif.get(j, (-np.inf, np.inf))

                # Define reasonable slider range, wider than individual tabs
                slider_min_c = max(lower_c, current_c - 10) if lower_c != -np.inf else current_c - 10
                slider_max_c = min(upper_c, current_c + 10) if upper_c != np.inf else current_c + 10
                if slider_max_c <= slider_min_c: slider_max_c = slider_min_c + 2 # Ensure range

                # Use session state to potentially persist slider values across reruns IF desired,
                # but typically sliders reset unless keys are carefully managed.
                # For simplicity here, we recalculate each time.
                new_c_whatif[j] = st.slider(
                    f"c{j + 1} (Range: [{lower_c:.2g}, {upper_c:.2g}])",
                    float(slider_min_c), float(slider_max_c), float(current_c),
                    step=0.1, key=f"whatif_c_{j}"
                )
                if not ((lower_c == -np.inf or new_c_whatif[j] >= lower_c - 1e-6) and \
                        (upper_c == np.inf or new_c_whatif[j] <= upper_c + 1e-6)):
                    any_outside_range = True
                    st.caption(f":warning: c{j+1} is outside allowable range!")


        # RHS Value Sliders
        with col2_wi:
            st.markdown("**RHS Values (b)**")
            rhs_sens_whatif = sensitivity.rhs_sensitivity_analysis()
            for i in range(solver.m):
                current_b = original_b[i]
                lower_b, upper_b = rhs_sens_whatif.get(i, (-np.inf, np.inf))

                 # Define reasonable slider range
                slider_min_b = max(lower_b, current_b - 10) if lower_b != -np.inf else current_b - 10
                slider_max_b = min(upper_b, current_b + 10) if upper_b != np.inf else current_b + 10
                if slider_max_b <= slider_min_b: slider_max_b = slider_min_b + 2

                new_b_whatif[i] = st.slider(
                    f"b{i + 1} (Range: [{lower_b:.2g}, {upper_b:.2g}])",
                    float(slider_min_b), float(slider_max_b), float(current_b),
                    step=0.1, key=f"whatif_b_{i}"
                )
                if not ((lower_b == -np.inf or new_b_whatif[i] >= lower_b - 1e-6) and \
                        (upper_b == np.inf or new_b_whatif[i] <= upper_b + 1e-6)):
                    any_outside_range = True
                    st.caption(f":warning: b{i+1} is outside allowable range!")


        st.markdown("---")

        if any_outside_range:
            st.warning("One or more parameters are outside their sensitivity ranges. The predicted objective value below is likely inaccurate. Re-solving is recommended.")

        # Calculate predicted new objective value
        shadow_prices_wi = sensitivity.shadow_prices()
        delta_obj_pred = 0

        # Effect of RHS changes
        for i in range(solver.m):
            if i < len(shadow_prices_wi): # Check index bounds
                delta_bi = new_b_whatif[i] - original_b[i]
                delta_obj_pred += shadow_prices_wi[i] * delta_bi

        # Effect of objective coefficient changes (on basic variables)
        # Need the optimal solution values
        optimal_sol_wi = sensitivity.optimal_solution
        for j in range(solver.n):
             if j < len(optimal_sol_wi): # Check index bounds
                 if optimal_sol_wi[j] > 1e-6: # Only for variables in the basis (positive value)
                     delta_cj = new_c_whatif[j] - original_c[j]
                     delta_obj_pred += delta_cj * optimal_sol_wi[j]

        predicted_obj = optimal_obj + delta_obj_pred

        col3_wi, col4_wi = st.columns(2)
        col3_wi.metric("Original Z*", f"{optimal_obj:.4f}")
        col4_wi.metric("Predicted New Z*", f"{predicted_obj:.4f}", delta=f"{predicted_obj - optimal_obj:.4f}")

        # Offer re-solve button
        if st.button("Re-solve with these What-If parameters", key="resolve_whatif"):
            st.session_state.sensitivity_resample = {
                'c': new_c_whatif.copy(),
                'A': original_A.copy(), # Keep A the same for now
                'b': new_b_whatif.copy()
            }
            st.session_state.auto_solve = True
            st.session_state.active_tab_index = 3 # Try to return to this tab
            st.session_state.coming_from_whatif = True # Signal for sidebar handling in app.py
            st.rerun() # Trigger app rerun

        # Visualization (only if 2D)
        if is_2d and plot_bounds:
            st.markdown("---")
            st.subheader("Combined Visualization (2D)")
            fig_combined = plot_combined_sensitivity(
                original_c, new_c_whatif, original_A, original_b, new_b_whatif,
                optimal_solution, plot_bounds
            )
            if fig_combined: st.plotly_chart(fig_combined, use_container_width=True)
        elif is_2d and not plot_bounds:
             st.warning("Could not determine plot bounds for combined visualization.")
        elif not is_2d:
            st.info("Combined visualization is available for 2D problems only.")

    # Store the index of the currently selected tab for the next run
    # This requires finding which tab object corresponds to which index
    # A simpler way if direct index isn't exposed is to check which context manager was entered last.
    # Or, explicitly check the selected attribute if available (might depend on Streamlit version)
    # A robust way: track it via a callback on the tabs if possible, or just update session state here.
    # Find the selected tab index
    current_selection = 0 # Default assumption
    # NOTE: Streamlit's `st.tabs` doesn't directly expose the selected index easily after creation.
    # We rely on the fact that the code inside the selected tab's `with` block runs.
    # We update the session state *after* the block for the currently selected tab finishes.
    # This requires checking which block is running. A bit hacky.
    # A cleaner approach might involve radio buttons instead of tabs if state persistence is critical.
    # Or, assume the order:
    if tabs[0].__enter__(): current_selection = 0; tabs[0].__exit__(None, None, None)
    if tabs[1].__enter__(): current_selection = 1; tabs[1].__exit__(None, None, None)
    if tabs[2].__enter__(): current_selection = 2; tabs[2].__exit__(None, None, None)
    if tabs[3].__enter__(): current_selection = 3; tabs[3].__exit__(None, None, None)
    # This approach is flawed because the 'with' blocks aren't mutually exclusive like this.

    # Simpler approach: just store the index based on which 'with' block is currently active.
    # This relies on Streamlit rendering the selected tab's content.
    # We can't directly know *which* tab is selected after the fact without more complex state handling.
    # For now, we set `active_tab_index` when the re-solve button is pressed.
    # Let's remove the automatic setting here as it's unreliable.
    # The state `active_tab_index` will be primarily set by the re-solve button logic.