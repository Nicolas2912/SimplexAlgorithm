import numpy as np
from scipy.optimize import linprog
from tabulate import tabulate
from fractions import Fraction


# Custom Exception Classes for better error handling
class SimplexError(Exception):
    """Base class for simplex-related errors."""
    pass

class InfeasibleProblemError(SimplexError):
    """Raised when the linear programming problem is infeasible."""
    pass

class UnboundedProblemError(SimplexError):
    """Raised when the linear programming problem is unbounded."""
    pass

class DegenerateSolutionError(SimplexError):
    """Raised when encountering degeneracy issues that cannot be resolved."""
    pass

class NumericalInstabilityError(SimplexError):
    """Raised when numerical instability is detected that may compromise results."""
    pass

class TableauCorruptionError(SimplexError):
    """Raised when the tableau appears to be in an invalid state."""
    pass


class PrimalSimplex:
    def __init__(self, c, A, b, use_fractions=False, fraction_digits=3, eq_constraints=False):
        """
        Initialize the simplex method for the problem:
        Minimize c^T x
        Subject to Ax <= b, x >= 0

        :param c: numpy array, coefficients of the objective function
        :param A: numpy matrix, coefficients of the constraints
        :param b: numpy array, right-hand side of the constraints
        :param use_fractions: bool, whether to print the tableau using fractions
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.m, self.n = A.shape
        self.rhs_negative = np.any(self.b < 0)
        self.use_fractions = use_fractions
        self.eq_constraints = eq_constraints
        self.original_c = np.array(c, dtype=float) if eq_constraints else None
        self.fraction_digits = fraction_digits
        self.phase = 1 if eq_constraints else 2
        self.in_phase_one = eq_constraints
        self.path_vertices = []

        # Validate inputs
        self._validate_inputs()

        self.tableau = self._initialize_tableau()
        
        # Store the initial vertex
        try:
            self._store_current_vertex()
        except Exception as e:
            import warnings
            warnings.warn(f"Could not store initial vertex: {e}. Path tracking may be incomplete.", UserWarning)
            self.path_vertices = []

    def _validate_inputs(self):
        """Validate all input parameters."""
        if self.m == 0:
            raise ValueError("Problem must have at least one constraint.")
        if self.n == 0:
            raise ValueError("Problem must have at least one variable.")
        
        # Check dimensions
        dimension_errors = []
        if len(self.b) != self.m:
            dimension_errors.append(f"Constraint vector b length ({len(self.b)}) does not match number of constraints ({self.m})")
        if len(self.c) != self.n:
            dimension_errors.append(f"Objective vector c length ({len(self.c)}) does not match number of variables ({self.n})")
        if self.A.shape != (self.m, self.n):
            dimension_errors.append(f"Constraint matrix A shape {self.A.shape} does not match expected ({self.m}, {self.n})")
        
        if dimension_errors:
            raise ValueError("; ".join(dimension_errors))
        
        # Check for invalid values
        for name, array in [("A", self.A), ("b", self.b), ("c", self.c)]:
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Array {name} contains non-finite values (NaN or Inf)")

    def get_basis_variable_indices(self):
        """
        Identifies the basic variable for each constraint row in the tableau.
        Returns dict mapping tableau row index to variable index.
        """
        basis_map = {}
        if not hasattr(self, 'tableau'):
            return basis_map

        rows, cols = self.tableau.shape
        num_vars_total = self.n + self.m

        for r in range(1, min(self.m + 1, rows)):
            for c in range(min(num_vars_total, cols - 1)):
                if (np.isclose(self.tableau[r, c], 1.0) and 
                    np.all(np.isclose(np.delete(self.tableau[:, c], r), 0.0))):
                    basis_map[r] = c
                    break
        return basis_map

    def _limit_fraction(self, value):
        """Limit the number of digits in a fraction's numerator and denominator."""
        if value is None or abs(float(value)) < 1e-10:
            return Fraction(0)
        
        try:
            frac = Fraction(value) if not isinstance(value, Fraction) else value
        except (TypeError, ValueError):
            return Fraction(0)
            
        max_value = 10 ** self.fraction_digits - 1
        n, d = frac.numerator, frac.denominator
        
        if abs(n) > max_value or abs(d) > max_value:
            return Fraction(float(frac)).limit_denominator(max_value)
        return frac

    def _initialize_tableau(self):
        """Initialize the simplex tableau."""
        self.original_A = self.A.copy()
        self.original_b = self.b.copy()

        if not self.eq_constraints:
            current_A = self.A.copy()
            current_b = self.b.copy()
            slack_vars = np.eye(self.m)

            # Handle negative RHS
            for i in range(self.m):
                if current_b[i] < 0:
                    current_A[i, :] *= -1
                    current_b[i] *= -1
                    slack_vars[i, i] = -1

            # Create tableau
            tableau = np.hstack((current_A, slack_vars))
            c_extended = np.hstack((self.c, np.zeros(self.m)))
            b_extended = current_b.reshape(-1, 1)

            tableau_main = np.vstack((-c_extended, tableau))
            tableau = np.hstack((tableau_main, np.vstack((0, b_extended))))

            return np.array(tableau, dtype=float)

        else:  # Equality constraints / Phase I
            self.artificial_indices = list(range(self.n, self.n + self.m))
            tableau = np.zeros((self.m + 1, self.n + self.m + 1), dtype=float)
            tableau[1:, :self.n] = self.A
            tableau[1:, self.n:self.n + self.m] = np.eye(self.m)
            tableau[1:, -1] = self.b
            tableau[0, self.artificial_indices] = -1

            # Make objective row consistent
            for i in range(1, self.m + 1):
                tableau[0, :] += tableau[i, :]

            return tableau

    def _store_current_vertex(self):
        """Extracts and stores the current vertex values."""
        vertex = np.zeros(self.n)
        try:
            basic_vars_info = self._get_basic_variables()
            for col_idx, row_idx in basic_vars_info:
                if col_idx < self.n:
                    vertex[col_idx] = max(0.0, self.tableau[row_idx, -1])
            
            # Avoid duplicates
            if not self.path_vertices or not np.allclose(vertex, self.path_vertices[-1], atol=1e-6):
                self.path_vertices.append(vertex.copy())
        except Exception as e:
            print(f"Warning: Could not extract vertex: {e}")

    def _find_pivot_column(self):
        """Find the entering variable using Bland's rule."""
        obj_coeffs = self.tableau[0, :-1]
        
        # Filter out artificial variables in Phase II
        if self.phase == 2 and self.eq_constraints and hasattr(self, 'artificial_indices'):
            mask = np.ones(obj_coeffs.shape, dtype=bool)
            for idx in self.artificial_indices:
                if idx < len(mask):
                    mask[idx] = False
            valid_indices = np.where(mask & (obj_coeffs > 1e-10))[0]
        else:
            valid_indices = np.where(obj_coeffs > 1e-10)[0]
            
        return min(valid_indices) if len(valid_indices) > 0 else None

    def _find_pivot_column_phase_one(self):
        """Find the entering variable for Phase I using Bland's rule."""
        obj_coeffs = self.tableau[0, :self.n + self.m]
        valid_indices = np.where(obj_coeffs > 1e-10)[0]
        return min(valid_indices) if len(valid_indices) > 0 else None

    def _find_pivot_row(self, pivot_col):
        """Find the leaving variable using minimum ratio test with Bland's rule."""
        valid_ratios = []

        for i in range(1, self.m + 1):
            pivot_col_entry = self.tableau[i, pivot_col]
            rhs_entry = self.tableau[i, -1]

            if pivot_col_entry > 1e-12:
                adjusted_rhs = max(0.0, rhs_entry) if rhs_entry >= -1e-12 else rhs_entry
                
                if adjusted_rhs >= 0:
                    ratio = adjusted_rhs / pivot_col_entry
                    basic_var_index = self._get_basic_variable_index_for_row(i)
                    valid_ratios.append((ratio, i, pivot_col_entry, basic_var_index))

        if not valid_ratios:
            return None

        # Sort by ratio, then by basic variable index (Bland's rule)
        valid_ratios.sort(key=lambda x: (x[0], x[3]))
        min_ratio, pivot_row, pivot_element, basic_var_idx = valid_ratios[0]
        
        if abs(pivot_element) < 1e-12:
            import warnings
            warnings.warn(f"Small pivot element {pivot_element:.2e} may cause numerical instability.", UserWarning)
            
        return pivot_row

    def _get_basic_variable_index_for_row(self, row_index):
        """Find the index of the basic variable for a given tableau row."""
        num_cols = self.tableau.shape[1] - 1
        
        for j in range(num_cols):
            if (abs(self.tableau[row_index, j] - 1.0) < 1e-10 and
                all(abs(self.tableau[i, j]) < 1e-10 for i in range(1, self.m + 1) if i != row_index)):
                return j
        
        return float('inf')

    def _pivot(self, pivot_row, pivot_col):
        """Perform the pivot operation with improved numerical stability."""
        pivot_element = self.tableau[pivot_row, pivot_col]
        
        if abs(pivot_element) < 1e-12:
            raise NumericalInstabilityError(
                f"Pivot element {pivot_element:.2e} at ({pivot_row}, {pivot_col}) is too small."
            )

        # Store original pivot row for better numerical accuracy
        pivot_row_original = self.tableau[pivot_row, :].copy()
        
        # Normalize pivot row
        self.tableau[pivot_row, :] /= pivot_element

        # Eliminate other entries in pivot column
        for i in range(self.m + 1):
            if i != pivot_row:
                factor = self.tableau[i, pivot_col]
                if abs(factor) > 1e-15:
                    self.tableau[i, :] -= (factor / pivot_element) * pivot_row_original
                    
        # Clean up numerical errors
        self.tableau[np.abs(self.tableau) < 1e-15] = 0.0
        self.tableau[:, pivot_col] = 0.0
        self.tableau[pivot_row, pivot_col] = 1.0

        self._store_current_vertex()

    def _phase_one(self):
        """Execute Phase I of the two-phase simplex method."""
        iteration = 0
        self.in_phase_one = True

        while True:
            pivot_col_idx = self._find_pivot_column_phase_one()
            if pivot_col_idx is None:
                break

            pivot_row = self._find_pivot_row(pivot_col_idx)
            if pivot_row is None:
                raise InfeasibleProblemError(
                    "Phase I detected unboundedness. Original problem is infeasible."
                )

            self._pivot(pivot_row, pivot_col_idx)
            iteration += 1

        # Check feasibility
        phase_one_obj_val = self.tableau[0, -1]
        if abs(phase_one_obj_val) > 1e-8:
            import warnings
            warnings.warn(f"Phase I objective value: {phase_one_obj_val:.2e}. Problem is infeasible.", UserWarning)
            return False

        # Check artificial variables
        basic_vars_info = self._get_basic_variables()
        for col_idx, row_idx in basic_vars_info:
            if col_idx in self.artificial_indices and abs(self.tableau[row_idx, -1]) > 1e-8:
                print(f"Warning: Artificial variable a{col_idx-self.n+1} has value {self.tableau[row_idx, -1]}")

        self.in_phase_one = False
        return True

    def _transition_to_phase_two(self):
        """Transition from Phase I to Phase II."""
        # Restore original objective function
        self.tableau[0, :] = 0.0
        self.tableau[0, :self.n] = -self.original_c

        # Make objective row consistent with current basis
        basic_vars_info = self._get_basic_variables()
        for col_idx, row_idx in basic_vars_info:
            obj_coeff = self.tableau[0, col_idx]
            if abs(obj_coeff) > 1e-10:
                self.tableau[0, :] -= obj_coeff * self.tableau[row_idx, :]

        # Try to pivot out artificial variables
        for col_idx, row_idx in basic_vars_info:
            if col_idx in self.artificial_indices:
                pivoted_out = False
                for j in range(self.n):
                    if abs(self.tableau[row_idx, j]) > 1e-10:
                        self._pivot(row_idx, j)
                        pivoted_out = True
                        break
                
                if not pivoted_out and abs(self.tableau[row_idx, -1]) > 1e-10:
                    print(f"Warning: Cannot pivot out artificial variable with value {self.tableau[row_idx, -1]}")

        self.phase = 2
        return True

    def _print_tableau(self, iteration):
        """Print the simplex tableau with consistent formatting."""
        print(f"\nIteration {iteration}:")
        
        # Create headers
        headers = [f"x{i + 1}" for i in range(self.n)]
        if self.in_phase_one:
            headers.extend([f"a{i + 1}" for i in range(self.m)])
        elif not self.eq_constraints:
            headers.extend([f"s{i + 1}" for i in range(self.m)])
        headers.append("RHS")

        # Create rows
        rows = []
        for i in range(self.m + 1):
            label = "z" if i == 0 else f"R{i}"
            display_data = list(self.tableau[i, :self.n])
            
            if self.in_phase_one or not self.eq_constraints:
                display_data.extend(self.tableau[i, self.n:self.n + self.m])
            
            display_data.append(self.tableau[i, -1])
            row_values = [f"{val:8.4f}" for val in display_data]
            rows.append([label] + row_values)

        try:
            print(tabulate(rows, headers=headers, colalign=["right"] * (len(headers) + 1), floatfmt=".4f"))
        except IndexError:
            print("Error printing tableau - header/data mismatch")

        basis = self._get_current_basis(self.tableau)
        print("Current Basis:", basis)

    def _get_variable_name(self, index):
        """Helper to get the formatted name of a variable by its column index."""
        if index < self.n:
            return f"x_{{{index + 1}}}"
        
        slack_or_art_index = index - self.n + 1
        if self.eq_constraints and getattr(self, 'in_phase_one', False):
            return f"a_{{{slack_or_art_index}}}"
        elif not self.eq_constraints:
            return f"s_{{{slack_or_art_index}}}"
        else:
            return f"Var_{{{index+1}}}"

    def _get_current_basis(self, current_tableau):
        """Identifies the basic variables from the current tableau state."""
        basis_vars = ["?"] * self.m
        num_var_cols = current_tableau.shape[1] - 1

        if current_tableau.shape[0] != self.m + 1:
            print(f"Warning: Tableau row count mismatch")

        for j in range(num_var_cols):
            col_data = current_tableau[1:, j]
            
            if len(col_data) != self.m:
                continue

            is_one = np.isclose(col_data, 1.0)
            is_zero = np.isclose(col_data, 0.0)

            if (np.sum(is_one) == 1 and np.sum(is_zero) == self.m - 1 and 
                np.isclose(np.sum(np.abs(col_data)), 1.0)):
                
                row_index = np.where(is_one)[0][0]
                if 0 <= row_index < self.m and basis_vars[row_index] == "?":
                    basis_vars[row_index] = self._get_variable_name(j)

        return basis_vars

    def _get_basic_variables(self):
        """Identify basic variables from the tableau by finding unit vectors."""
        basic_vars = []
        used_rows = set()
        num_cols = self.tableau.shape[1] - 1

        for j in range(num_cols):
            col = self.tableau[:, j]
            constraint_rows = col[1:]
            rows_one = np.where(np.isclose(constraint_rows, 1.0))[0] + 1
            rows_nonzero_constraints = np.where(np.abs(constraint_rows) > 1e-10)[0] + 1

            if len(rows_one) == 1 and len(rows_nonzero_constraints) == 1:
                row_with_one = rows_one[0]
                obj_entry = col[0]
                is_obj_zero = abs(obj_entry) <= 1e-10
                
                accept_variable = (is_obj_zero or 
                                 getattr(self, 'in_phase_one', False) or 
                                 getattr(self, 'eq_constraints', False))
                
                if accept_variable and row_with_one not in used_rows:
                    basic_vars.append((j, row_with_one))
                    used_rows.add(row_with_one)

        return sorted(basic_vars, key=lambda x: x[1])

    def _check_tableau_integrity(self):
        """Check tableau for corruption (NaN/Inf values)."""
        if not np.all(np.isfinite(self.tableau)):
            non_finite_locations = np.where(~np.isfinite(self.tableau))
            rows, cols = non_finite_locations
            if len(rows) > 0:
                first_bad_row, first_bad_col = rows[0], cols[0]
                bad_value = self.tableau[first_bad_row, first_bad_col]
                raise TableauCorruptionError(
                    f"Tableau corruption: non-finite value {bad_value} at ({first_bad_row}, {first_bad_col}). "
                    f"Total {len(rows)} corrupted entries."
                )

    def solve(self):
        """Solve the linear programming problem using the simplex method."""
        self._check_tableau_integrity()

        if self.eq_constraints:
            if not self._phase_one():
                raise InfeasibleProblemError("Phase I failed to find feasible solution.")
            
            self._transition_to_phase_two()
            self._store_current_vertex()

        # Phase II (or main phase for <= constraints)
        iteration = 0
        max_iterations = max(1000, self.m * self.n * 10)
        tableau_history = []
        cycling_check_interval = 10
        
        while iteration < max_iterations:
            pivot_col = self._find_pivot_column()
            if pivot_col is None:
                break

            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row is None:
                raise UnboundedProblemError("Problem is unbounded.")

            # Cycling detection
            if iteration % cycling_check_interval == 0 and iteration > 0:
                try:
                    current_signature = tuple(np.round(self.tableau.flatten(), 8))
                    if current_signature in tableau_history:
                        import warnings
                        warnings.warn(f"Potential cycling detected at iteration {iteration}.", UserWarning)
                        break
                    tableau_history.append(current_signature)
                    
                    if len(tableau_history) > 20:
                        tableau_history.pop(0)
                except (MemoryError, OverflowError) as e:
                    import warnings
                    warnings.warn(f"Disabled cycling detection: {e}", UserWarning)
                    cycling_check_interval = max_iterations

            self._pivot(pivot_row, pivot_col)
            
            try:
                self._check_tableau_integrity()
            except TableauCorruptionError:
                raise TableauCorruptionError(
                    f"Tableau corruption after pivot at iteration {iteration}. "
                    f"Pivot: row {pivot_row}, column {pivot_col}."
                )
            
            iteration += 1

        if iteration >= max_iterations:
            import warnings
            warnings.warn(f"Maximum iterations ({max_iterations}) reached without convergence.", UserWarning)

        # Extract final solution
        solution = np.zeros(self.n)
        try:
            basic_vars_info = self._get_basic_variables()
            
            if len(basic_vars_info) < self.m:
                import warnings
                warnings.warn(f"Found only {len(basic_vars_info)} basic variables, expected {self.m}.", UserWarning)
            
            for col_idx, row_idx in basic_vars_info:
                val = self.tableau[row_idx, -1]
                if not np.isfinite(val):
                    var_name = f"x_{col_idx+1}" if col_idx < self.n else f"slack/artificial_{col_idx}"
                    raise TableauCorruptionError(f"Non-finite value {val} in {var_name}")
                
                if col_idx < self.n:
                    solution[col_idx] = max(0.0, val)
                    
        except Exception as e:
            if isinstance(e, (SimplexError, TableauCorruptionError)):
                raise
            else:
                raise TableauCorruptionError(f"Failed to extract solution: {e}")

        solution = np.clip(solution, 0.0, None)
        solution[np.abs(solution) < 1e-12] = 0.0
        optimal_value = -self.tableau[0, -1]

        return solution, optimal_value


class SensitivityAnalysis:
    """Class for performing sensitivity analysis on the optimal solution."""

    def __init__(self, simplex_solver):
        """Initialize with a solved PrimalSimplex instance."""
        if not isinstance(simplex_solver, PrimalSimplex):
            raise TypeError(f"Expected PrimalSimplex instance, got {type(simplex_solver).__name__}")
            
        self.solver = simplex_solver
        if not hasattr(self.solver, 'tableau') or self.solver.tableau is None:
            raise ValueError("Solver must have valid tableau. Call solve() first.")
             
        # Check if optimal
        obj_coeffs = self.solver.tableau[0, :-1]
        if np.any(obj_coeffs > 1e-8):
            import warnings
            warnings.warn("Sensitivity analysis on potentially non-optimal tableau.", UserWarning)

        # Initialize analysis data
        try:
            self.basis_indices = self._get_basis_indices()
            self.nonbasis_indices = [i for i in range(self.solver.tableau.shape[1] - 1) 
                                   if i not in self.basis_indices]
            self.A_B_inv = self._get_basis_inverse()
            self.optimal_obj, self.optimal_solution = self._get_optimal_solution()
        except Exception as e:
            raise ValueError(f"Failed to initialize sensitivity analysis: {e}")

    def _get_basis_indices(self):
        """Extract indices of basic variables from the optimal tableau."""
        try:
            basic_vars_info = self.solver._get_basic_variables()
            basis_col_indices = [col_idx for col_idx, row_idx in basic_vars_info]
            
            if len(basis_col_indices) != self.solver.m:
                print(f"Warning: Found {len(basis_col_indices)} basis indices, expected {self.solver.m}.")
                
            if len(basis_col_indices) == 0:
                raise ValueError("No basic variables found - tableau may be corrupted")
                    
            return basis_col_indices
            
        except Exception as e:
            raise ValueError(f"Failed to extract basis indices: {e}")

    def _get_basis_inverse(self):
        """Calculate the inverse of the basis matrix."""
        original_A_ext = np.hstack((self.solver.original_A, np.eye(self.solver.m)))
        valid_basis_indices = [idx for idx in self.basis_indices if idx < original_A_ext.shape[1]]
        
        if len(valid_basis_indices) != self.solver.m:
            raise ValueError(f"Cannot form basis matrix. Found {len(valid_basis_indices)} valid indices, expected {self.solver.m}.")

        basis_matrix_B = original_A_ext[:, valid_basis_indices]

        if basis_matrix_B.shape[0] != basis_matrix_B.shape[1]:
            raise ValueError(f"Basis matrix not square: {basis_matrix_B.shape}")

        try:
            return np.linalg.inv(basis_matrix_B)
        except np.linalg.LinAlgError:
            raise ValueError("Basis matrix is singular, cannot compute inverse.")

    def _get_optimal_solution(self):
        """Extract the optimal solution and objective value."""
        if not hasattr(self.solver, 'tableau') or self.solver.tableau is None:
            raise ValueError("Solver tableau not available.")

        obj_value = -self.solver.tableau[0, -1]
        solution = np.zeros(self.solver.n)
        
        try:
            basic_vars_info = self.solver._get_basic_variables()
            for col_idx, row_idx in basic_vars_info:
                if col_idx < self.solver.n:
                    try:
                        val = float(self.solver.tableau[row_idx, -1])
                        solution[col_idx] = max(0.0, val)
                    except (TypeError, ValueError):
                        solution[col_idx] = 0.0
        except Exception as e:
            print(f"Warning: Error extracting basic variables: {e}")
            solution.fill(np.nan)

        return obj_value, solution

    def rhs_sensitivity_analysis(self):
        """Perform sensitivity analysis on the right-hand side values."""
        sensitivity_ranges = {}
        current_basic_values = self.A_B_inv @ self.solver.original_b

        for r in range(self.solver.m):
            e_r = np.zeros(self.solver.m)
            e_r[r] = 1.0
            A_B_inv_e_r = self.A_B_inv @ e_r

            current_rhs = self.solver.original_b[r]
            allowable_decrease = allowable_increase = float('inf')

            for i in range(self.solver.m):
                if abs(current_basic_values[i]) < 1e-9:
                    continue
                    
                coeff = A_B_inv_e_r[i]
                if coeff > 1e-10:
                    allowable_decrease = min(allowable_decrease, current_basic_values[i] / coeff)
                elif coeff < -1e-10:
                    allowable_increase = min(allowable_increase, -current_basic_values[i] / coeff)

            lower_bound = current_rhs - allowable_decrease if allowable_decrease != float('inf') else -np.inf
            upper_bound = current_rhs + allowable_increase if allowable_increase != float('inf') else np.inf
            sensitivity_ranges[r] = (lower_bound, upper_bound)

        return sensitivity_ranges

    def objective_sensitivity_analysis(self):
        """Perform sensitivity analysis on the objective function coefficients."""
        sensitivity_ranges = {}
        reduced_costs = self.solver.tableau[0, :-1].copy()
        original_c = self.solver.c
        original_A_ext = np.hstack((self.solver.original_A, np.eye(self.solver.m)))
        original_c_ext = np.hstack((original_c, np.zeros(self.solver.m)))
        valid_basis_indices = [idx for idx in self.basis_indices if idx < len(original_c_ext)]
        C_B = original_c_ext[valid_basis_indices]

        for j in range(self.solver.n):
            current_c_j = original_c[j]
            allowable_decrease = allowable_increase = float('inf')

            if j not in self.basis_indices:  # Non-basic variable
                reduced_cost_min = -reduced_costs[j]
                allowable_decrease = reduced_cost_min
                allowable_increase = float('inf')
                lower_bound = current_c_j - allowable_decrease
                upper_bound = current_c_j + allowable_increase

            else:  # Basic variable
                basis_row_in_tableau = -1
                basic_vars_info = self.solver._get_basic_variables()
                for b_col, b_row in basic_vars_info:
                    if b_col == j:
                        basis_row_in_tableau = b_row
                        break
                        
                if basis_row_in_tableau == -1:
                    continue

                for k in self.nonbasis_indices:
                    reduced_cost_k_min = -reduced_costs[k]
                    y_jk = self.solver.tableau[basis_row_in_tableau, k]

                    if abs(y_jk) > 1e-10:
                        ratio = reduced_cost_k_min / y_jk
                        if y_jk > 0:
                            allowable_decrease = min(allowable_decrease, ratio)
                        else:
                            allowable_increase = min(allowable_increase, -ratio)

                lower_bound = current_c_j - allowable_decrease if allowable_decrease != float('inf') else -np.inf
                upper_bound = current_c_j + allowable_increase if allowable_increase != float('inf') else np.inf

            sensitivity_ranges[j] = (lower_bound, upper_bound)

        return sensitivity_ranges

    def shadow_prices(self):
        """Calculate shadow prices (dual variables) for each constraint."""
        shadow_prices = np.zeros(self.solver.m)
        for i in range(self.solver.m):
            slack_idx = self.solver.n + i
            if slack_idx < self.solver.tableau.shape[1] - 1:
                shadow_prices[i] = -self.solver.tableau[0, slack_idx]
            else:
                print(f"Warning: Slack variable s{i+1} column index {slack_idx} out of bounds.")
        return shadow_prices

    def format_range(self, range_tuple, var_value=None):
        """Format a sensitivity range in a readable way."""
        lower, upper = range_tuple
        lower_str = "-∞" if lower == -np.inf else f"{lower:.4f}"
        upper_str = "+∞" if upper == np.inf else f"{upper:.4f}"
        
        if var_value is not None:
            delta_lower = "any decrease" if lower == -np.inf else f"{var_value - lower:.4f}"
            delta_upper = "any increase" if upper == np.inf else f"{upper - var_value:.4f}"
            return f"[{lower_str}, {upper_str}] (Current: {var_value:.4f}, Δ-: {delta_lower}, Δ+: {delta_upper})"
        else:
            return f"[{lower_str}, {upper_str}]"


def add_sensitivity_analysis_to_simplex(PrimalSimplex):
    """Extend the PrimalSimplex class with sensitivity analysis methods."""
    PrimalSimplex.perform_sensitivity_analysis = lambda self: SensitivityAnalysis(self)
    return PrimalSimplex


def solve_lp_scipy(c, A, b, constraint_type='<='):
    """
    Solve a linear programming problem using SciPy's linprog function:
    Minimize c^T x
    Subject to A x [constraint_type] b, x >= 0
    """
    c, A, b = np.asarray(c, dtype=float), np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    
    if A.shape[0] != len(b) or A.shape[1] != len(c):
        raise ValueError("Inconsistent dimensions in problem specification")
    
    if constraint_type == '<=':
        result = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * len(c), method='highs')
    elif constraint_type == '=':
        result = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * len(c), method='highs')
    else:
        raise ValueError("constraint_type must be either '<=' or '='")
    
    if result.success: 
        return result.x, result.fun
    else: 
        error_messages = {
            2: "Problem is infeasible",
            3: "Problem is unbounded"
        }
        msg = error_messages.get(result.status, f"SciPy linprog failed: {result.message} (Status: {result.status})")
        raise ValueError(msg)


# Example Usage
if __name__ == "__main__":
    """
    Example problem:

        Minimize:    z = -2x₁ - 3x₂ - 4x₃

        Subject to:
            x₁ + x₂ + x₃      ≤ 6
            2x₁ + x₂          ≤ 4
            x₂ + 3x₃          ≤ 7
            xⱼ ≥ 0   for j = 1, 2, 3
    """

    c = np.array([-2, -3, -4])
    A = np.array([
        [1, 1, 1],
        [2, 1, 0],
        [0, 1, 3]
    ])
    b = np.array([6, 4, 7])

    solver = PrimalSimplex(c, A, b)
    solution, optimal_value = solver.solve()
    print(f"Optimal solution: {solution}")
    print(f"Optimal value: {optimal_value}")
    
