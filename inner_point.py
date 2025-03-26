import numpy as np
import sys
import plotly.graph_objects as go # Import plotly

# ============================================
# KarmarkarSolver Class (Keep the previous correct version here)
# ============================================
class KarmarkarSolver:
    """
    Implements Karmarkar's Inner-Point Algorithm structure for Linear Programs
    in Karmarkar Form.
    (Includes path storage for visualization).
    """

    def __init__(self, A: np.ndarray, c: np.ndarray, sense: str = 'minimize',
                 alpha: float = 0.5,
                 conv_tol: float = 1e-8,
                 max_iterations: int = 100):
        """ Initializes the KarmarkarSolver. """
        self.A = np.asarray(A)
        self.c = np.asarray(c).reshape(-1, 1)

        # --- Input validation remains the same ---
        if self.A.ndim != 2: raise ValueError("A must be a 2D matrix.")
        if self.c.ndim != 2 or self.c.shape[1] != 1: raise ValueError("c must be effectively a column vector.")
        if self.A.shape[1] != self.c.shape[0]: raise ValueError(f"Dimension mismatch: A ({self.A.shape}) and c ({self.c.shape}).")
        if not (0 < alpha < 1): raise ValueError("alpha must be strictly between 0 and 1.")
        if sense not in ['maximize', 'minimize']: raise ValueError("sense must be 'maximize' or 'minimize'.")
        # --- End Input Validation ---

        self.m, self.n = self.A.shape
        self.sense = sense
        self.sense_factor = -1.0 if sense == 'minimize' else 1.0
        self.alpha = alpha
        self.conv_tol = conv_tol
        self.max_iterations = max_iterations

        if self.n <= 1: raise ValueError("Number of variables (n) must be greater than 1.")
        self.r = 1.0 / np.sqrt(self.n * (self.n - 1))
        self.e = np.ones((self.n, 1))
        self._machine_eps = sys.float_info.epsilon

        # Path storage
        self.path = [] # Initialize list to store points

        print(f"Initialized Karmarkar-style Solver:")
        print(f"  Objective Sense: {self.sense}")
        print(f"  n (variables): {self.n}, m (constraints): {self.m}")
        print(f"  alpha: {self.alpha}, Convergence Tol: {self.conv_tol:.2e}")
        print(f"  r: {self.r:.6f}, max_iterations: {self.max_iterations}")

    def _check_inner_point(self, x: np.ndarray):
        """ Checks if x is a feasible inner point. """
        # --- Check code remains the same ---
        if not np.all(x > self._machine_eps * 10):
            min_val = np.min(x)
            raise ValueError(f"Initial point x0 is not strictly an inner point (min element {min_val} is too close to zero).")
        if not np.isclose(np.sum(x), 1.0):
            raise ValueError(f"Initial point x0 does not sum to 1 (sum={np.sum(x)}).")
        if not np.allclose(self.A @ x, 0, atol=1e-8):
            constraint_check = (self.A @ x).flatten()
            raise ValueError(f"Initial point x0 does not satisfy Ax = 0 (Ax={constraint_check}).")
        print("Initial point x0 is verified as a feasible inner point.")


    def solve(self, x0: np.ndarray):
        """
        Executes the Karmarkar-style algorithm and stores the path.

        Returns:
            tuple: (x_final, iterations, final_obj, stop_reason, path_list)
        """
        x_k = np.asarray(x0).reshape(-1, 1)
        if x_k.shape != (self.n, 1):
            raise ValueError(f"Initial point x0 must have shape ({self.n}, 1)")

        self._check_inner_point(x_k)
        self.path = [x_k.copy().flatten()] # Store initial point (flattened)

        y_center = self.e / self.n
        prev_obj_val = np.inf
        stop_reason = "Max iterations reached"

        print("\nStarting Karmarkar-style Iterations...")
        for k in range(self.max_iterations):
            obj_val = (self.c.T @ x_k).item()
            obj_change = abs(obj_val - prev_obj_val)
            rel_obj_change = obj_change / (abs(prev_obj_val) + 1e-10)
            print(f"Iter {k}: Objective = {obj_val:.8f}  (Change: {obj_change:.4e}, Rel Change: {rel_obj_change:.4e})")

            # --- Step 1: Check Stopping Criteria ---
            stop_criterion_met = False
            if k > 0:
                 if obj_change < self.conv_tol:
                     stop_criterion_met = True
                     stop_reason = f"Objective change ({obj_change:.4e}) < tolerance ({self.conv_tol:.4e})"

            if stop_criterion_met:
                print(f"\nStopping criterion met: {stop_reason}")
                self.path.append(x_k.copy().flatten()) # Store final point
                return x_k, k, obj_val, stop_reason, self.path

            prev_obj_val = obj_val

            # --- Step 2: Transformation ---
            D_k = np.diag(x_k.flatten())

            # --- Step 3: Projected Gradient ---
            try:
                A_tilde = self.A @ D_k
                B = np.vstack((A_tilde, self.e.T))
                P = np.eye(self.n) - B.T @ np.linalg.pinv(B @ B.T) @ B
            except np.linalg.LinAlgError as e:
                 print(f"\nError: numpy.linalg.LinAlgError: {e}")
                 raise RuntimeError("Linear algebra error during projection calculation.") from e

            c_tilde = D_k @ self.c
            c_p = P @ c_tilde
            norm_cp = np.linalg.norm(c_p)

            # Criterion 2: Projected gradient norm small
            if norm_cp < self.conv_tol:
                stop_reason = f"Projected gradient norm ({norm_cp:.4e}) < tolerance ({self.conv_tol:.4e})"
                print(f"\nStopping: {stop_reason}")
                self.path.append(x_k.copy().flatten()) # Store final point
                return x_k, k, obj_val, stop_reason, self.path

            # --- Step 3b: Update in Transformed Space ---
            step_direction = self.sense_factor * (c_p / norm_cp)
            y_k_plus_1 = y_center + self.alpha * self.r * step_direction

            # --- Step 4: Inverse Transformation ---
            numerator = D_k @ y_k_plus_1
            denominator = self.e.T @ numerator

            if np.isclose(denominator.item(), 0):
                 print("\nError: Denominator in inverse transformation is close to zero.")
                 raise RuntimeError("Numerical issue in inverse transformation.")

            x_k_plus_1 = numerator / denominator.item()

            # Safeguard: Ensure positivity
            min_x_next = np.min(x_k_plus_1)
            if min_x_next <= self._machine_eps * 10 :
                 print(f"\nWarning: Component of x_k became too close to zero ({min_x_next=:.4e}).")
                 stop_reason = "Potential loss of strict interior point property"
                 print(f"Stopping: {stop_reason}")
                 self.path.append(x_k_plus_1.copy().flatten()) # Store final point
                 return x_k_plus_1, k + 1, (self.c.T @ x_k_plus_1).item(), stop_reason, self.path

            # Update x_k and store path
            x_k = x_k_plus_1
            self.path.append(x_k.copy().flatten())

        # --- End Loop ---
        print(f"\nStopping: {stop_reason}")
        if k == self.max_iterations - 1: # Ensure last point stored if max iter reached
             self.path.append(x_k.copy().flatten())

        return x_k, self.max_iterations, (self.c.T @ x_k).item(), stop_reason, self.path

# ============================================
# NEW Plotly Visualization Function
# ============================================
def visualize_path_plotly(path_points, x_known_opt=None):
    """
    Visualizes the path of interior points in 3D using Plotly.
    (Revised to avoid dense marker overlap).
    """
    if not path_points:
        print("No path points to visualize.")
        return
    if len(path_points[0]) != 3:
        print("Visualization requires 3D points.")
        return

    path_array = np.array(path_points)
    x1 = path_array[:, 0]
    x2 = path_array[:, 1]
    x3 = path_array[:, 2]

    hover_texts = [f"Iter {i}<br>x1={pt[0]:.4f}<br>x2={pt[1]:.4f}<br>x3={pt[2]:.4f}"
                   for i, pt in enumerate(path_points)]

    fig = go.Figure()

    # 1. Feasible Region (Line segment)
    feasible_x = np.array([0, 1/3])
    feasible_y = np.array([1/3, 0])
    feasible_z = np.array([2/3, 2/3])
    fig.add_trace(go.Scatter3d(
        x=feasible_x, y=feasible_y, z=feasible_z,
        mode='lines+markers', # Keep markers for endpoints here
        line=dict(color='grey', width=5, dash='dash'), # Make line thicker
        marker=dict(size=5, color='grey'), # Markers for endpoints
        name='Feasible Region'
    ))

    # 2. Solver Path (Line ONLY) - <<< CHANGE HERE <<<
    fig.add_trace(go.Scatter3d(
        x=x1, y=x2, z=x3,
        mode='lines', # Use 'lines' ONLY, no intermediate markers
        line=dict(color='blue', width=3), # Make line slightly thicker
        hoverinfo='text', # Hover shows info for line segments
        hovertext=hover_texts,
        name='Solver Path'
    ))

    # --- Markers for specific points ---

    # 3. Start Point
    fig.add_trace(go.Scatter3d(
        x=[x1[0]], y=[x2[0]], z=[x3[0]],
        mode='markers',
        marker=dict(size=8, color='lime', symbol='circle', line=dict(color='black', width=1)),
        name=f'Start Point (x0)',
        hoverinfo='text',
        hovertext=hover_texts[0]
    ))

    # 4. Final Point
    fig.add_trace(go.Scatter3d(
        x=[x1[-1]], y=[x2[-1]], z=[x3[-1]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='circle', line=dict(color='black', width=1)),
        name=f'Final Point (Iter {len(path_points)-1})',
        hoverinfo='text',
        hovertext=hover_texts[-1]
    ))

    # 5. Known Optimum
    if x_known_opt is not None:
        fig.add_trace(go.Scatter3d(
            x=[x_known_opt[0]], y=[x_known_opt[1]], z=[x_known_opt[2]],
            mode='markers',
            marker=dict(size=10, color='gold', symbol='diamond', line=dict(color='black', width=1)), # Using diamond
            name='Known Optimum',
            hoverinfo='text',
            hovertext=f"Known Opt<br>x1={x_known_opt[0]:.4f}<br>x2={x_known_opt[1]:.4f}<br>x3={x_known_opt[2]:.4f}"
        ))

    # --- Layout ---
    fig.update_layout(
        title="Karmarkar Algorithm Path Visualization (Interactive)",
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='x3',
            aspectratio=dict(x=1, y=1, z=0.5), # Adjust aspect ratio if needed
            camera_eye=dict(x=1.5, y=-1.5, z=0.75) # Adjust initial camera view
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text='Legend'
    )

    fig.show()


# --- Example Usage (Calling Plotly function) ---
if __name__ == "__main__":
    print("--- Running Karmarkar Example (Problem from Image - With Plotly Visualization) ---")

    c_problem = np.array([[-1.0], [0.0], [0.0]])
    A_problem = np.array([[2.0, 2.0, -1.0]])
    alpha_param = 0.5
    conv_tol_param = 1e-8
    max_iter_param = 100
    x0_problem = np.array([[1/6], [1/6], [2/3]])
    x_known_opt_vis = np.array([1/3, 0, 2/3]) # Flattened

    try:
        solver = KarmarkarSolver(A=A_problem, c=c_problem, sense='minimize',
                                 alpha=alpha_param, conv_tol=conv_tol_param,
                                 max_iterations=max_iter_param)

        x_approx, iterations, final_obj, reason, path_data = solver.solve(x0=x0_problem)

        # --- Print Results (same as before) ---
        print("\n--- Results ---")
        print(f"Algorithm finished after {iterations} iterations.")
        print(f"Stopping Reason: {reason}")
        print(f"Approximate Optimal Solution (x): \n{x_approx}")
        print(f"Final Objective Value (c^T x): {final_obj:.8f}")
        print(f"(Note: The true optimal value is -1/3 ≈ -0.33333333)")
        print("\nConstraint Verification:")
        print(f"  Ax = 0: {(A_problem @ x_approx).flatten()} (Should be close to [0])")
        print(f"  1^T x = 1: {np.sum(x_approx):.8f} (Should be close to 1)")
        min_x_val = np.min(x_approx)
        print(f"  x >= 0: {min_x_val >= -1e-9} (Min element: {min_x_val:.4e})")


        # --- Visualization using Plotly ---
        print("\nVisualizing the path using Plotly...")
        visualize_path_plotly(path_data, x_known_opt=x_known_opt_vis)


    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)