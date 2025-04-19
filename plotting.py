# plotting.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import convert_to_fraction # Optional: if needed for labels

# --- 2D Plotting (Matplotlib - kept for compatibility/alternative) ---

def plot_lp_problem_2d(c, A, b, solution=None, path_vertices=None, title="LP Feasible Region (2D)", min_x=-1, max_x=10, min_y=-1, max_y=10):
    """
    Plot the LP problem, feasible region, optimal solution, and simplex path in 2D using Matplotlib.
    Args:
        c, A, b: Problem parameters (only first 2 variables used).
        solution: Optimal solution [x1, x2].
        path_vertices: List of [x1, x2] points for simplex path.
        title: Plot title.
        min_x, max_x, min_y, max_y: Initial plot bounds.
    Returns: Matplotlib figure or None, error message string or None.
    """
    try:
        if len(c) < 2 or A.shape[1] < 2:
            return None, "Need at least 2 variables for 2D plot."

        c_2d = np.array(c[:2], dtype=float)
        A_2d = np.array(A[:, :2], dtype=float)
        b_2d = np.array(b, dtype=float)

        # --- Determine Plot Bounds ---
        plot_min_x, plot_max_x = min_x, max_x
        plot_min_y, plot_max_y = min_y, max_y
        points_x, points_y = [0], [0] # Include origin

        if solution is not None and len(solution) >= 2:
            points_x.append(solution[0])
            points_y.append(solution[1])
        if path_vertices:
            for v in path_vertices:
                if len(v) >= 2:
                    points_x.append(v[0])
                    points_y.append(v[1])

        # Add constraint intercepts (if positive)
        for i in range(A_2d.shape[0]):
            if abs(A_2d[i, 0]) > 1e-9:
                intercept_x = b_2d[i] / A_2d[i, 0]
                if intercept_x >= 0: points_x.append(intercept_x)
            if abs(A_2d[i, 1]) > 1e-9:
                intercept_y = b_2d[i] / A_2d[i, 1]
                if intercept_y >= 0: points_y.append(intercept_y)

        if len(points_x) > 1 and len(points_y) > 1:
            data_min_x, data_max_x = min(points_x), max(points_x)
            data_min_y, data_max_y = min(points_y), max(points_y)
            x_margin = max(1, (data_max_x - data_min_x) * 0.2)
            y_margin = max(1, (data_max_y - data_min_y) * 0.2)
            plot_min_x = min(plot_min_x, data_min_x - x_margin)
            plot_max_x = max(plot_max_x, data_max_x + x_margin)
            plot_min_y = min(plot_min_y, data_min_y - y_margin)
            plot_max_y = max(plot_max_y, data_max_y + y_margin)

        # Ensure minimum range and non-negativity if applicable
        plot_min_x = min(0, plot_min_x)
        plot_min_y = min(0, plot_min_y)
        if plot_max_x - plot_min_x < 1: plot_max_x = plot_min_x + 1
        if plot_max_y - plot_min_y < 1: plot_max_y = plot_min_y + 1

        fig, ax = plt.subplots(figsize=(7, 4))
        x_grid = np.linspace(plot_min_x, plot_max_x, 400)
        y_grid = np.linspace(plot_min_y, plot_max_y, 400)
        X, Y = np.meshgrid(x_grid, y_grid)

        # --- Plot Constraints ---
        colors = plt.cm.viridis(np.linspace(0, 1, A_2d.shape[0]))
        for i in range(A_2d.shape[0]):
            a1, a2 = A_2d[i, 0], A_2d[i, 1]
            bi = b_2d[i]
            label = f'C{i+1}: {a1:.2g}x₁ + {a2:.2g}x₂ ≤ {bi:.2g}'
            color = colors[i]

            if abs(a2) > 1e-9:
                constraint_y = (bi - a1 * x_grid) / a2
                ax.plot(x_grid, constraint_y, color=color, label=label, alpha=0.8)
            elif abs(a1) > 1e-9:
                constraint_x = bi / a1
                if plot_min_x <= constraint_x <= plot_max_x:
                    ax.axvline(x=constraint_x, color=color, label=label, alpha=0.8)

        # --- Fill Feasible Region using contourf ---
        mask = (X >= -1e-9) & (Y >= -1e-9) # Non-negativity with tolerance
        for i in range(A_2d.shape[0]):
            mask &= (A_2d[i, 0] * X + A_2d[i, 1] * Y <= b_2d[i] + 1e-9) # Constraints with tolerance

        # Convert boolean mask to float (0.0 for False, 1.0 for True)
        feasible_data = mask.astype(float)

        # Fill the area where feasible_data is essentially 1.0
        # levels=[0.5, 1.5] isolates the region where mask was True
        ax.contourf(X, Y, feasible_data, levels=[0.5, 1.5], colors=['lightblue'], alpha=0.4, zorder=0)
        # zorder=0 ensures the fill is drawn behind lines/points

        # --- Objective Function Contours ---
        if not np.all(np.isclose(c_2d, 0)):
            Z = c_2d[0] * X + c_2d[1] * Y
            z_min, z_max = np.min(Z), np.max(Z)
            if not np.isclose(z_min, z_max):
                 levels = np.linspace(z_min, z_max, 10)
                 CS = ax.contour(X, Y, Z, levels=levels, colors='grey', linestyles='dashed', alpha=0.6)
                 ax.clabel(CS, inline=True, fontsize=8, fmt='%.1f')

                 # Improvement Direction Arrow (for Minimization)
                 mid_x = (plot_min_x + plot_max_x) / 2
                 mid_y = (plot_min_y + plot_max_y) / 2
                 arrow_len = min(plot_max_x - plot_min_x, plot_max_y - plot_min_y) / 12
                 grad_norm = np.linalg.norm(c_2d)
                 if grad_norm > 1e-9:
                     dx = -c_2d[0] / grad_norm * arrow_len
                     dy = -c_2d[1] / grad_norm * arrow_len
                     arrow_start_x = mid_x
                     arrow_start_y = mid_y
                     in_bounds_start = (plot_min_x <= arrow_start_x <= plot_max_x) and (plot_min_y <= arrow_start_y <= plot_max_y)
                     in_bounds_end = (plot_min_x <= (arrow_start_x + dx) <= plot_max_x) and (plot_min_y <= (arrow_start_y + dy) <= plot_max_y)
                     # Only plot if endpoints are somewhat reasonable
                     if in_bounds_start or in_bounds_end:
                          ax.arrow(arrow_start_x, arrow_start_y, dx, dy,
                                 head_width=arrow_len * 0.3, head_length=arrow_len * 0.5,
                                 fc='black', ec='black', zorder=5)

        # --- Plot Solution and Path ---
        if path_vertices:
            path_x = [v[0] for v in path_vertices if len(v)>=2]
            path_y = [v[1] for v in path_vertices if len(v)>=2]
            if len(path_x) > 1:
                ax.plot(path_x, path_y, 'o-', color='purple', markersize=5, linewidth=1.5, label='Simplex Path', zorder=5)
                for i, (px, py) in enumerate(zip(path_x, path_y)):
                    ax.annotate(f'{i}', (px, py), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, color='purple')

        # --- Optimal Solution ---
        if solution is not None and len(solution) >= 2:
            sol_x, sol_y = solution[0], solution[1]
            if plot_min_x <= sol_x <= plot_max_x and plot_min_y <= sol_y <= plot_max_y:
                 ax.scatter(sol_x, sol_y, color='red', s=100, marker='*', label=f'Optimal: ({sol_x:.3g}, {sol_y:.3g})', zorder=10)
            else: # Indicate if outside view
                 ax.text(0.98, 0.02, f'Optimal outside view\n({sol_x:.3g}, {sol_y:.3g})',
                         ha='right', va='bottom', transform=ax.transAxes, fontsize=8, color='red', style='italic')

        # --- Add proxy plot for the arrow legend ---
        # This plots nothing visible but provides the correct legend entry
        if not np.all(np.isclose(c_2d, 0)) and grad_norm > 1e-9:
             ax.plot([], [], color='black', linewidth=1.5, label='Improvement Direction')

        # --- Plot Configuration ---
        ax.set_xlim(plot_min_x, plot_max_x)
        ax.set_ylim(plot_min_y, plot_max_y)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=0) # Ensure grid is behind axes
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(title)

        # --- Add Thicker Black Axes Lines (x=0 and y=0) ---
        axis_linewidth = 1.5 # Adjust thickness as needed
        axis_zorder = 0.5    # Place axes above grid but below data/constraints

        # Draw y-axis (x=0) if visible
        if plot_min_x <= 0 <= plot_max_x:
            ax.axvline(0, color='black', linewidth=axis_linewidth, zorder=axis_zorder)
        # Draw x-axis (y=0) if visible
        if plot_min_y <= 0 <= plot_max_y:
            ax.axhline(0, color='black', linewidth=axis_linewidth, zorder=axis_zorder)
        # --------------------------------------------------

        # --- Legend Handling ---
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # Place legend below plot, centered, horizontal arrangement
        ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fontsize=8, ncol=3, frameon=False) # frameon=False for cleaner look below plot

        plt.subplots_adjust(bottom=0.25) # Make space below the plot for the legend

        return fig, None # Return figure

    except Exception as e:
        import traceback
        # Use st context if available, otherwise print
        try:
            import streamlit as st
            st.error(f"Error creating 2D plot: {e}\n{traceback.format_exc()}") # Show full error in Streamlit
        except ImportError:
            print(f"Error creating 2D plot: {e}\n{traceback.format_exc()}")
        return None, f"Error creating 2D plot: {str(e)}"


# --- 3D Plotting (Plotly) ---

def plot_lp_problem_3d(c, A, b, solution=None, path_vertices=None, title="LP Feasible Region (3D)"):
    """
    Create a 3D visualization of the LP problem using Plotly.
    Args:
        c, A, b: Problem parameters (first 3 variables used).
        solution: Optimal solution [x1, x2, x3].
        path_vertices: List of [x1, x2, x3] points for simplex path.
        title: Plot title.
    Returns: Plotly figure or None.
    """
    if len(c) < 3 or A.shape[1] < 3:
        st.warning("3D Visualization requires 3 variables.")
        return None

    c_3d = np.array(c[:3], dtype=float)
    A_3d = np.array(A[:, :3], dtype=float)
    b_3d = np.array(b, dtype=float)

    fig = go.Figure()

    # --- Determine Bounds ---
    max_coord = 5.0 # Default starting bound
    all_coords = [0] # Include origin
    if solution is not None and len(solution) >= 3: all_coords.extend(solution[:3])
    if path_vertices:
        for v in path_vertices:
            if len(v) >= 3: all_coords.extend(v[:3])

    # Add constraint intercepts (if positive and finite)
    for i in range(A_3d.shape[0]):
        for j in range(3):
            if abs(A_3d[i, j]) > 1e-9:
                intercept = b_3d[i] / A_3d[i, j]
                if np.isfinite(intercept) and intercept >= -1e-9: # Allow close to zero
                    all_coords.append(intercept)

    if all_coords:
         max_val = max(c for c in all_coords if np.isfinite(c) and c >=0) # Max non-negative coord
         max_coord = max(max_coord, max_val * 1.2) # Add buffer

    # --- Grid for Isosurfaces ---
    grid_res = 15 # Resolution (keep modest for performance)
    linsp = np.linspace(0, max_coord, grid_res)
    X, Y, Z = np.meshgrid(linsp, linsp, linsp, indexing='ij')

    # --- Plot Constraint Planes ---
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(A_3d.shape[0]):
        a1, a2, a3 = A_3d[i, 0], A_3d[i, 1], A_3d[i, 2]
        rhs = b_3d[i]
        constraint_values = a1 * X + a2 * Y + a3 * Z
        label = f'C{i+1}: {a1:.1f}x₁+ {a2:.1f}x₂+ {a3:.1f}x₃ ≤ {rhs:.1f}'
        color = colors[i % len(colors)]
        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=constraint_values.flatten(),
            isomin=rhs, isomax=rhs, # Show the boundary plane
            surface_count=1,
            opacity=0.3,
            colorscale=[[0, color], [1, color]], # Single color
            showscale=False,
            name=label,
            hoverinfo='name'
        ))

    # --- Plot Simplex Path ---
    if path_vertices:
        path_x = [v[0] for v in path_vertices if len(v)>=3]
        path_y = [v[1] for v in path_vertices if len(v)>=3]
        path_z = [v[2] for v in path_vertices if len(v)>=3]
        if len(path_x) > 0:
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines+markers+text', # Add text labels
                line=dict(color='purple', width=4),
                marker=dict(color='purple', size=5),
                text=[str(i) for i in range(len(path_x))], # Iteration numbers
                textposition='top center',
                name='Simplex Path'
            ))

    # --- Plot Optimal Solution ---
    if solution is not None and len(solution) >= 3:
        x_sol, y_sol, z_sol = solution[0], solution[1], solution[2]
        obj_value = np.dot(c_3d, [x_sol, y_sol, z_sol]) if len(c_3d)==3 else np.nan
        fig.add_trace(go.Scatter3d(
            x=[x_sol], y=[y_sol], z=[z_sol],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name=f'Optimal ({x_sol:.2f}, {y_sol:.2f}, {z_sol:.2f})<br>Z={obj_value:.2f}', # Use <br> for newline in hover
            hoverinfo='name'
        ))

    # --- Configure Layout ---
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x₁', yaxis_title='x₂', zaxis_title='x₃',
            xaxis=dict(range=[0, max_coord], autorange=False, zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.5)'),
            yaxis=dict(range=[0, max_coord], autorange=False, zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.5)'),
            zaxis=dict(range=[0, max_coord], autorange=False, zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0,0,0,0.5)'),
            aspectmode='cube', # 'cube' looks better for LP
            camera_eye=dict(x=1.5, y=1.5, z=1.5) # Adjust initial view
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.6)'),
        margin=dict(l=10, r=10, b=10, t=50) # Add some margin
    )
    return fig


# --- Sensitivity Plotting (Plotly - specific to sensitivity tabs) ---

def plot_objective_sensitivity(original_c, new_c, optimal_solution, changed_var_index, original_obj, new_obj, plot_bounds):
    """
    Plotly: Compare original vs new objective function contours (2D).
    """
    if len(original_c) < 2 or len(new_c) < 2 or len(optimal_solution) < 2:
        st.warning("Cannot plot objective sensitivity without 2 variables.")
        return None

    var_name = f"c{changed_var_index + 1}"
    title = f"Effect of Changing {var_name} from {original_c[changed_var_index]:.2f} to {new_c[changed_var_index]:.2f}"

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Original Objective", "New Objective"],
                        shared_xaxes=True, shared_yaxes=True)

    x_min, x_max, y_min, y_max = plot_bounds
    x_range = np.linspace(x_min, x_max, 50) # Reduced resolution for speed
    y_range = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x_range, y_range)

    Z_original = original_c[0] * X + original_c[1] * Y
    Z_new = new_c[0] * X + new_c[1] * Y

    # Original Contour
    fig.add_trace(go.Contour(z=Z_original, x=x_range, y=y_range, colorscale='Blues', showscale=False,
                             contours=dict(coloring='lines', showlabels=True, labelfont=dict(size=9))),
                  row=1, col=1)
    # New Contour
    fig.add_trace(go.Contour(z=Z_new, x=x_range, y=y_range, colorscale='Reds', showscale=False,
                             contours=dict(coloring='lines', showlabels=True, labelfont=dict(size=9))),
                  row=1, col=2)

    # Optimal Point (same for both if basis doesn't change)
    sol_x, sol_y = optimal_solution[0], optimal_solution[1]
    fig.add_trace(go.Scatter(x=[sol_x], y=[sol_y], mode='markers', marker=dict(color='green', size=10, symbol='star'), name='Optimal Point'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[sol_x], y=[sol_y], mode='markers', marker=dict(color='green', size=10, symbol='star'), showlegend=False), row=1, col=2) # No duplicate legend

    # Annotations for objective values near the optimal point
    fig.add_annotation(x=sol_x, y=sol_y, text=f" Orig. Z*={original_obj:.2f}", showarrow=True, arrowhead=1, ax=20, ay=-30, row=1, col=1)
    fig.add_annotation(x=sol_x, y=sol_y, text=f" New Z*={new_obj:.2f}", showarrow=True, arrowhead=1, ax=20, ay=-30, row=1, col=2)

    fig.update_layout(title_text=title, height=450, margin=dict(l=20, r=20, t=80, b=20))
    fig.update_xaxes(title_text="x₁", range=[x_min, x_max], row=1, col=1)
    fig.update_xaxes(title_text="x₁", range=[x_min, x_max], row=1, col=2)
    fig.update_yaxes(title_text="x₂", range=[y_min, y_max], row=1, col=1)

    return fig

def plot_rhs_sensitivity(A, original_b, new_b, optimal_solution, changed_constraint_index, plot_bounds):
    """
    Plotly: Show effect of changing one RHS value on constraints (2D).
    """
    if A.shape[1] < 2 or len(optimal_solution) < 2:
        st.warning("Cannot plot RHS sensitivity without 2 variables.")
        return None

    A_2d = A[:, :2]
    constr_name = f"b{changed_constraint_index + 1}"
    title = f"Effect of Changing {constr_name} from {original_b[changed_constraint_index]:.2f} to {new_b[changed_constraint_index]:.2f}"

    fig = go.Figure()

    x_min, x_max, y_min, y_max = plot_bounds
    x_range = np.array([x_min, x_max]) # Need only endpoints for lines

    # Plot original constraints
    for i in range(A_2d.shape[0]):
        a1, a2 = A_2d[i, 0], A_2d[i, 1]
        bi = original_b[i]
        is_changed = (i == changed_constraint_index)
        line_style = 'solid' if is_changed else 'dash'
        color = 'blue'
        label = f'Orig C{i+1}'

        if abs(a2) > 1e-9:
            y_vals = (bi - a1 * x_range) / a2
            fig.add_trace(go.Scatter(x=x_range, y=y_vals, mode='lines', line=dict(color=color, width=2, dash=line_style), name=label))
        elif abs(a1) > 1e-9:
            x_val = bi / a1
            fig.add_trace(go.Scatter(x=[x_val, x_val], y=[y_min, y_max], mode='lines', line=dict(color=color, width=2, dash=line_style), name=label))

    # Plot the new changed constraint
    i = changed_constraint_index
    a1, a2 = A_2d[i, 0], A_2d[i, 1]
    bi_new = new_b[i]
    color = 'red'
    label = f'New C{i+1}'
    if abs(a2) > 1e-9:
        y_vals = (bi_new - a1 * x_range) / a2
        fig.add_trace(go.Scatter(x=x_range, y=y_vals, mode='lines', line=dict(color=color, width=2.5), name=label))
    elif abs(a1) > 1e-9:
        x_val = bi_new / a1
        fig.add_trace(go.Scatter(x=[x_val, x_val], y=[y_min, y_max], mode='lines', line=dict(color=color, width=2.5), name=label))

    # Plot Optimal Point
    sol_x, sol_y = optimal_solution[0], optimal_solution[1]
    fig.add_trace(go.Scatter(x=[sol_x], y=[sol_y], mode='markers', marker=dict(color='green', size=10, symbol='star'), name='Optimal Point'))

    # Add non-negativity axes if within bounds
    if 0 >= x_min: fig.add_trace(go.Scatter(x=[0, 0], y=[y_min, y_max], mode='lines', line=dict(color='black', width=1), showlegend=False))
    if 0 >= y_min: fig.add_trace(go.Scatter(x=[x_min, x_max], y=[0, 0], mode='lines', line=dict(color='black', width=1), showlegend=False))


    fig.update_layout(title_text=title, height=450, margin=dict(l=20, r=20, t=80, b=20))
    fig.update_xaxes(title_text="x₁", range=[x_min, x_max])
    fig.update_yaxes(title_text="x₂", range=[y_min, y_max])
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))

    return fig


def plot_combined_sensitivity(original_c, new_c, A, original_b, new_b, optimal_solution, plot_bounds):
    """
    Plotly: Side-by-side comparison of original vs modified problem (2D).
    """
    if A.shape[1] < 2 or len(optimal_solution) < 2:
        st.warning("Cannot plot combined sensitivity without 2 variables.")
        return None

    A_2d = A[:, :2]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Original Problem", "Modified Problem"],
                        shared_xaxes=True, shared_yaxes=True)

    x_min, x_max, y_min, y_max = plot_bounds
    x_range_cont = np.linspace(x_min, x_max, 50) # For contours
    y_range_cont = np.linspace(y_min, y_max, 50)
    X_cont, Y_cont = np.meshgrid(x_range_cont, y_range_cont)
    x_range_line = np.array([x_min, x_max]) # For lines
    Z_original = original_c[0] * X_cont + original_c[1] * Y_cont
    Z_new = new_c[0] * X_cont + new_c[1] * Y_cont

    # --- Check for Constant Z data ---
    # Check if Z varies enough to draw contours. Use a tolerance.
    z_orig_range = np.ptp(Z_original) # Peak-to-peak range
    z_new_range = np.ptp(Z_new)
    tolerance = 1e-6 # Adjust if necessary
    is_Z_original_constant = z_orig_range < tolerance
    is_Z_new_constant = z_new_range < tolerance

    # --- Plot 1: Original Problem ---
    # Objective Contour (only if not constant)
    if not is_Z_original_constant:
        try:
            fig.add_trace(go.Contour(z=Z_original, x=x_range_cont, y=y_range_cont, colorscale='Blues', showscale=False,
                                     name="Orig. Obj.", legendgroup="obj",
                                     contours=dict(coloring='lines', showlabels=True, labelfont=dict(size=9))),
                          row=1, col=1)
        except Exception as e_contour:
            # Warn user if contour plotting fails, but continue
            st.warning(f"Note: Could not plot original objective contour. Error: {e_contour}")
    elif np.any(np.abs(original_c[:2]) > tolerance): # Check if original C was non-zero
        # Add annotation if data was constant over this view
         fig.add_annotation(text="Orig. Obj. constant in view", xref="paper", yref="paper",
                            x=0.25, y=0.5, showarrow=False, row=1, col=1, bgcolor="rgba(255,255,255,0.5)")


    # Constraints (keep as before, with showlegend=False)
    for i in range(A_2d.shape[0]):
        a1, a2 = A_2d[i, 0], A_2d[i, 1]
        bi = original_b[i]
        label = f'Orig C{i+1}'
        if abs(a2) > 1e-9:
            y_vals = (bi - a1 * x_range_line) / a2
            fig.add_trace(go.Scatter(x=x_range_line, y=y_vals, mode='lines', line=dict(color='blue', width=2), name=label, legendgroup="orig_constr", showlegend=False), row=1, col=1)
        elif abs(a1) > 1e-9:
            x_val = bi / a1
            fig.add_trace(go.Scatter(x=[x_val, x_val], y=[y_min, y_max], mode='lines', line=dict(color='blue', width=2), name=label, legendgroup="orig_constr", showlegend=False), row=1, col=1)

    # --- Plot 2: Modified Problem ---
    # Objective Contour (only if not constant)
    if not is_Z_new_constant:
        try:
            fig.add_trace(go.Contour(z=Z_new, x=x_range_cont, y=y_range_cont, colorscale='Reds', showscale=False,
                                     name="New Obj.", legendgroup="obj",
                                     contours=dict(coloring='lines', showlabels=True, labelfont=dict(size=9))),
                          row=1, col=2)
        except Exception as e_contour:
             # Warn user if contour plotting fails, but continue
             st.warning(f"Note: Could not plot new objective contour. Error: {e_contour}")
    elif np.any(np.abs(new_c[:2]) > tolerance): # Check if new C was non-zero
        # Add annotation if data was constant over this view
        fig.add_annotation(text="New Obj. constant in view", xref="paper", yref="paper",
                           x=0.75, y=0.5, showarrow=False, row=1, col=2, bgcolor="rgba(255,255,255,0.5)")


    # Constraints (keep as before, with showlegend=False)
    for i in range(A_2d.shape[0]):
        a1, a2 = A_2d[i, 0], A_2d[i, 1]
        bi_new = new_b[i]
        label = f'New C{i+1}'
        if abs(a2) > 1e-9:
            y_vals = (bi_new - a1 * x_range_line) / a2
            fig.add_trace(go.Scatter(x=x_range_line, y=y_vals, mode='lines', line=dict(color='red', width=2), name=label, legendgroup="new_constr", showlegend=False), row=1, col=2)
        elif abs(a1) > 1e-9:
            x_val = bi_new / a1
            fig.add_trace(go.Scatter(x=[x_val, x_val], y=[y_min, y_max], mode='lines', line=dict(color='red', width=2), name=label, legendgroup="new_constr", showlegend=False), row=1, col=2)


    # --- Optimal Point (keep as is) ---
    sol_x, sol_y = optimal_solution[0], optimal_solution[1]
    fig.add_trace(go.Scatter(x=[sol_x], y=[sol_y], mode='markers', marker=dict(color='green', size=10, symbol='star'), name='Optimal Point', legendgroup="opt"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[sol_x], y=[sol_y], mode='markers', marker=dict(color='green', size=10, symbol='star'), showlegend=False, legendgroup="opt"), row=1, col=2)

    # --- Layout (keep as is) ---
    fig.update_layout(title_text="Original vs. Modified Problem Parameters", height=450, margin=dict(l=20, r=20, t=80, b=20))
    fig.update_xaxes(title_text="x₁", range=[x_min, x_max], row=1, col=1)
    fig.update_xaxes(title_text="x₁", range=[x_min, x_max], row=1, col=2)
    fig.update_yaxes(title_text="x₂", range=[y_min, y_max], row=1, col=1)

    return fig


def get_plot_bounds(A, b, solution=None, path_vertices=None, default_max=10):
    """
    Calculate reasonable plot bounds for 2D plots based on solution, path, and intercepts.
    Returns (min_x, max_x, min_y, max_y)
    """
    min_x_bound, max_x_bound = 0, default_max
    min_y_bound, max_y_bound = 0, default_max
    points_x, points_y = [0], [0] # Include origin

    if solution is not None and len(solution) >= 2:
        points_x.append(solution[0])
        points_y.append(solution[1])
    if path_vertices:
        for v in path_vertices:
            if len(v) >= 2:
                points_x.append(v[0])
                points_y.append(v[1])

    # Add constraint intercepts (if positive and finite)
    A_2d = A[:, :2]
    b_2d = b
    for i in range(A_2d.shape[0]):
        # X-intercept
        if abs(A_2d[i, 0]) > 1e-9:
            intercept_x = b_2d[i] / A_2d[i, 0]
            if np.isfinite(intercept_x) and intercept_x >= -1e-9: points_x.append(intercept_x)
        # Y-intercept
        if abs(A_2d[i, 1]) > 1e-9:
            intercept_y = b_2d[i] / A_2d[i, 1]
            if np.isfinite(intercept_y) and intercept_y >= -1e-9: points_y.append(intercept_y)

    # Filter out non-finite values before calculating min/max
    finite_points_x = [p for p in points_x if np.isfinite(p)]
    finite_points_y = [p for p in points_y if np.isfinite(p)]

    if finite_points_x:
        data_min_x, data_max_x = min(finite_points_x), max(finite_points_x)
        x_margin = max(1, (data_max_x - data_min_x) * 0.2) # 20% margin, min 1
        min_x_bound = data_min_x - x_margin
        max_x_bound = data_max_x + x_margin

    if finite_points_y:
        data_min_y, data_max_y = min(finite_points_y), max(finite_points_y)
        y_margin = max(1, (data_max_y - data_min_y) * 0.2) # 20% margin, min 1
        min_y_bound = data_min_y - y_margin
        max_y_bound = data_max_y + y_margin

    # Ensure bounds are reasonable (e.g., include origin)
    final_min_x = min(0, min_x_bound)
    final_min_y = min(0, min_y_bound)
    final_max_x = max(default_max, max_x_bound)
    final_max_y = max(default_max, max_y_bound)

    # Ensure minimum range
    if final_max_x - final_min_x < 2: final_max_x = final_min_x + 2
    if final_max_y - final_min_y < 2: final_max_y = final_min_y + 2

    return final_min_x, final_max_x, final_min_y, final_max_y


# Placeholder functions for large-scale 2D plots if needed (can reuse plot_lp_problem_2d with different bounds)
# def plot_full_problem_2d(...): ...
# def plot_solution_focus_2d(...): ...