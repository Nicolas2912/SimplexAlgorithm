import numpy as np
from scipy.optimize import linprog
from tabulate import tabulate
from fractions import Fraction


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
        self.c = c
        self.A = A
        self.b = b
        self.m, self.n = A.shape
        self.rhs_negative = self._check_negative_rhs()

        self.use_fractions = use_fractions
        self.eq_constraints = eq_constraints
        self.tableau = self._initialize_tableau()
        self.fraction_digits = fraction_digits

        self.phase = 1 if eq_constraints else 2  # Track which phase we're in

    def _check_negative_rhs(self):
        """
        Check if any of the constraints have negative RHS values.
        """
        for i in range(self.m):
            if self.b[i] < 0:
                return True
        return False

    def _limit_fraction(self, value):
        """
        Limit the number of digits in a fraction's numerator and denominator.

        :param value: numeric value to convert and limit
        :return: Fraction with limited digits
        """
        # Handle None, zero, and very small numbers
        if value is None:
            return Fraction(0)
        if abs(float(value)) < 1e-10:
            return Fraction(0)

        # Convert to Fraction if not already
        if not isinstance(value, Fraction):
            try:
                frac = Fraction(value).limit_denominator()
            except (TypeError, ValueError):
                return Fraction(0)
        else:
            frac = value

        max_value = 10 ** self.fraction_digits - 1
        n, d = frac.numerator, frac.denominator

        # If either numerator or denominator exceeds the digit limit
        if abs(n) > max_value or abs(d) > max_value:
            # Convert to float and create a new fraction with limited denominator
            float_val = float(frac)
            return Fraction(float_val).limit_denominator(max_value)

        return frac

    def _initialize_tableau(self):
        """
        Initialize the simplex tableau properly handling negative RHS values.
        """
        # Store original problem
        self.original_A = self.A.copy()
        self.original_b = self.b.copy()

        if not self.eq_constraints:
            # Process constraints with negative RHS
            slack_vars = np.eye(self.m)
            rhs_negative = False

            for i in range(self.m):
                if self.b[i] < 0:
                    # Multiply row by -1 and use negative slack
                    self.A[i, :] *= -1
                    self.b[i] *= -1
                    slack_vars[i, i] = -1
                    rhs_negative = True  # Use negative slack for transformed constraints
                else:
                    slack_vars[i, i] = 1

            # multiply the objective function by -1
            if rhs_negative:
                self.c *= -1
            # Create tableau
            tableau = np.hstack((self.A, slack_vars))
            c_extended = np.hstack((self.c, np.zeros(self.m)))
            b_extended = self.b.reshape(-1, 1)
            tableau = np.vstack((c_extended, tableau))
            tableau = np.hstack((tableau, np.vstack((0, b_extended))))

            return tableau
        else:
            # Store original objective coefficients
            self.original_c = self.c.copy()

            # Add artificial variables for equality constraints
            artificial_vars = np.eye(self.m)
            self.artificial_indices = list(range(self.n, self.n + self.m))

            # Construct initial tableau
            tableau = np.zeros((self.m + 1, self.n + self.m + 1))

            # Set constraint coefficients
            tableau[1:, :self.n] = self.A
            tableau[1:, self.n:self.n + self.m] = artificial_vars
            tableau[1:, -1] = self.b

            # Set up Phase I objective: minimize sum of artificial variables
            tableau[0, self.artificial_indices] = 1

            # Make the objective row consistent with the artificial variable basis
            for i in range(1, self.m + 1):
                tableau[0, :] -= tableau[i, :]

            return tableau

    def _find_pivot_column(self):
        """
        Find the entering variable (pivot column) by choosing the most negative coefficient.
        """
        # Get the objective row coefficients for the original variables
        obj_coeffs = self.tableau[0, :-1]

        # Find all negative coefficients
        neg_indices = np.where(obj_coeffs < -1e-10)[0]

        if len(neg_indices) == 0:
            return None  # No negative coefficients, we're optimal

        # Find the most negative coefficient
        most_neg_idx = neg_indices[np.argmin(obj_coeffs[neg_indices])]
        return most_neg_idx

    def _find_pivot_row(self, pivot_col):
        """
        Find the leaving variable (pivot row) using the minimum ratio test.
        """
        ratios = []
        for i in range(1, self.m + 1):
            if self.tableau[i, pivot_col] > 1e-10:
                ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                ratios.append((ratio, i))

        if not ratios:
            return None

        return min(ratios, key=lambda x: x[0])[1]

    def _pivot(self, pivot_row, pivot_col):
        """
        Perform the pivot operation to update the tableau.
        """
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_element
        for i in range(self.m + 1):
            if i != pivot_row:
                self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]

    def _phase_one(self):
        """
        Execute Phase I of the two-phase simplex method.
        Returns True if a feasible solution is found, False otherwise.
        """
        iteration = 0
        self.in_phase_one = True

        while True:
            self._print_tableau(iteration)

            # Find entering variable (most negative coefficient in objective row)
            pivot_col = np.argmin(self.tableau[0, :-1])
            if self.tableau[0, pivot_col] >= -1e-10:
                break  # Phase I complete

            # Find leaving variable
            ratios = []
            for i in range(1, self.m + 1):
                if self.tableau[i, pivot_col] > 1e-10:
                    ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                    ratios.append(ratio)
                else:
                    ratios.append(np.inf)

            if all(r == np.inf for r in ratios):
                return False  # Problem is infeasible

            pivot_row = 1 + np.argmin(ratios)

            # Perform pivot
            print(f"\nPivot: Entering Column = {pivot_col + 1}, Leaving Row = {pivot_row}")
            self._pivot(pivot_row, pivot_col)
            iteration += 1

        # Check if all artificial variables are zero
        if abs(self.tableau[0, -1]) > 1e-10:
            return False  # Problem is infeasible

        return True

    def _transition_to_phase_two(self):
        """
        Transition from Phase I to Phase II.
        Sets up the tableau for Phase II with the original objective function.
        """
        # Set up the original objective row
        self.tableau[0, :] = 0
        self.tableau[0, :self.n] = self.original_c

        # Make the objective row consistent with the current basis
        basic_vars = self._get_basic_variables()
        for col_idx, row_idx in basic_vars:
            if col_idx < self.n:  # Only adjust for original variables
                self.tableau[0, :] -= self.tableau[0, col_idx] * self.tableau[row_idx, :]

        # Remove artificial columns (but keep basic artificials if needed)
        cols_to_keep = list(range(self.n))  # Keep original variables

        # Check which artificial variables are in the basis and still needed
        for j in self.artificial_indices:
            col = self.tableau[:, j]
            is_basic = False
            for i in range(1, self.m + 1):
                if abs(col[i] - 1.0) < 1e-10 and all(abs(col[k]) < 1e-10 for k in range(1, self.m + 1) if k != i):
                    is_basic = True
                    break
            if is_basic:
                cols_to_keep.append(j)

        # Keep RHS column
        cols_to_keep.append(self.tableau.shape[1] - 1)

        # Update tableau
        self.tableau = self.tableau[:, cols_to_keep]
        self.phase = 2
        return True

    def _print_tableau(self, iteration):
        """
        Print the simplex tableau with consistent column headers and reduced costs.
        Dynamically adjusts display based on phase and number of variables.
        """
        print(f"\nIteration {iteration}:")

        if self.phase == 1:
            # Phase I display
            headers = (
                    [f"x{i + 1}" for i in range(self.n)] +  # Original variables
                    [f"a{i + 1}" for i in range(self.m)] +  # Artificial variables
                    ["RHS"]
            )

            # Create rows with proper spacing
            rows = []
            row_label_width = 2  # Width for "--" or "Rx"
            for i in range(self.m + 1):
                label = "z" if i == 0 else f"R{i}"
                row_data = self.tableau[i, :]
                # Convert row data to list and ensure proper formatting
                row_values = [f"{val:8.4f}" for val in row_data]
                rows.append([label] + row_values)

        else:
            # Phase II display
            # For equality constraints after Phase I, only show original variables
            if self.eq_constraints:
                headers = [f"x{i + 1}" for i in range(self.n)] + ["RHS"]
                num_cols = self.n + 1  # Original variables + RHS
            else:
                # For inequality constraints, include slack variables
                headers = ([f"x{i + 1}" for i in range(self.n)] +
                           [f"s{i + 1}" for i in range(self.m)] +
                           ["RHS"])
                num_cols = self.n + self.m + 1  # Original + slack + RHS

            # Create rows with proper spacing
            rows = []
            row_label_width = 2  # Width for "--" or "Rx"
            for i in range(self.m + 1):
                label = "z" if i == 0 else f"R{i}"
                # Only take the columns we need based on the phase
                row_data = self.tableau[i, :num_cols]
                # Convert row data to list and ensure proper formatting
                row_values = [f"{val:8.4f}" for val in row_data]
                rows.append([label] + row_values)

        # Use tabulate with proper formatting
        print(tabulate(rows,
                       headers=headers,
                       colalign=["right"] * (len(headers) + 1),  # +1 for row labels
                       floatfmt=".4f"))

        # Print the current basis
        basis = self._get_current_basis()
        print("Current Basis:", basis)

    def _get_current_basis(self):
        """Helper method to get current basis variables"""
        basis = []
        num_cols = self.tableau.shape[1] - 1  # Exclude RHS column
        for i in range(num_cols):
            col = self.tableau[:, i]
            if np.sum(col == 1) == 1 and np.sum(col == 0) == self.m:
                if i < self.n:
                    basis.append(f"x{i + 1}")
                elif not self.eq_constraints:
                    basis.append(f"s{i + 1 - self.n}")
                elif self.in_phase_one:
                    basis.append(f"a{i + 1 - self.n}")
        return basis

    def _get_basic_variables(self):
        """
        Identify basic variables from the tableau by finding unit vectors.
        Returns list of (column_index, row_index) pairs sorted by row_index.
        """
        basic_vars = []
        used_rows = set()

        for j in range(self.tableau.shape[1] - 1):  # Exclude RHS column
            col = self.tableau[:, j]
            nonzero_indices = np.where(abs(col) > 1e-10)[0]

            # Check if column has exactly one 1 and rest zeros
            if (len(nonzero_indices) == 1 and
                    abs(col[nonzero_indices[0]] - 1.0) < 1e-10 and
                    nonzero_indices[0] not in used_rows):
                row_idx = nonzero_indices[0]
                basic_vars.append((j, row_idx))
                used_rows.add(row_idx)

        return sorted(basic_vars, key=lambda x: x[1])

    def solve(self):
        """
        Solve the linear programming problem using the two-phase simplex method.
        """
        if self.eq_constraints:
            # Execute Phase I
            print("\nStarting Phase I")
            if not self._phase_one():
                raise Exception("Problem is infeasible")

            print("\nStarting Phase II")
            self._transition_to_phase_two()

        # Phase II
        iteration = 0
        while True:
            self._print_tableau(iteration)

            # Find entering variable
            pivot_col = self._find_pivot_column()
            if pivot_col is None:
                break  # Optimal solution found

            # Find leaving variable
            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row is None:
                raise Exception("Problem is unbounded")

            # Perform pivot
            print(f"\nPivot: Entering Column = {pivot_col + 1}, Leaving Row = {pivot_row}")
            self._pivot(pivot_row, pivot_col)
            iteration += 1

        # Extract solution
        solution = np.zeros(self.n)
        basic_vars = self._get_basic_variables()

        for col_idx, row_idx in basic_vars:
            if col_idx < self.n:  # Only original variables
                solution[col_idx] = self.tableau[row_idx, -1]

        if self.rhs_negative:
            optimal_value = self.tableau[0, -1]
        else:
            optimal_value = -self.tableau[0, -1]
        return solution, optimal_value


class SensitivityAnalysis:
    """
    Class for performing sensitivity analysis on the optimal solution of a linear program.
    Follows the mathematical formulation from the lecture script.
    """

    def __init__(self, simplex_solver):
        """
        Initialize with a solved PrimalSimplex instance.

        Parameters:
        -----------
        simplex_solver : PrimalSimplex
            A solved simplex instance with an optimal solution.
        """
        self.solver = simplex_solver

        # Verify that the simplex solver has found an optimal solution
        if not hasattr(self.solver, 'tableau'):
            raise ValueError("Simplex solver must have a tableau")

        # Get optimal basis information
        self.basis_indices = self._get_basis_indices()
        self.nonbasis_indices = [i for i in range(self.solver.n + self.solver.m)
                                 if i not in self.basis_indices]

        # Get optimal basis inverse
        self.A_B_inv = self._get_basis_inverse()

        # Get the optimal objective value and solution
        self.optimal_obj, self.optimal_solution = self._get_optimal_solution()

    def _get_basis_indices(self):
        """Extract indices of basic variables from the optimal tableau."""
        basis = []
        # Identify columns that form the basis
        for j in range(self.solver.tableau.shape[1] - 1):  # Exclude RHS column
            col = self.solver.tableau[:, j]
            # Check if this is a unit vector (corresponding to a basic variable)
            if np.sum(col == 1) == 1 and np.sum(col == 0) == self.solver.m:
                # Find the row where 1 appears
                row_idx = np.where(col == 1)[0][0]
                if row_idx > 0:  # Skip the objective row
                    basis.append(j)

        return basis

    def _get_basis_inverse(self):
        """Calculate the inverse of the basis matrix."""
        # Extract the basis columns from the constraint matrix
        basis_cols = []
        for idx in self.basis_indices:
            if idx < self.solver.n:  # Original variable
                basis_cols.append(self.solver.A[:, idx])
            else:  # Slack variable
                slack_idx = idx - self.solver.n
                slack_col = np.zeros(self.solver.m)
                slack_col[slack_idx] = 1
                basis_cols.append(slack_col)

        basis_matrix = np.column_stack(basis_cols)

        # Calculate the inverse
        return np.linalg.inv(basis_matrix)

    def _get_optimal_solution(self):
        """Extract the optimal solution and objective value."""
        # Get the objective value (negative of the value in the bottom-right cell of tableau)
        obj_value = -self.solver.tableau[0, -1]

        # Extract the solution vector
        solution = np.zeros(self.solver.n)
        for i, idx in enumerate(self.basis_indices):
            if idx < self.solver.n:  # Only for original variables
                # Get the value from the RHS column
                row = np.where(self.solver.tableau[:, idx] == 1)[0][0]
                solution[idx] = self.solver.tableau[row, -1]

        return obj_value, solution

    def rhs_sensitivity_analysis(self):
        """
        Perform sensitivity analysis on the right-hand side values.

        Returns:
        --------
        dict: Dictionary mapping constraint indices to (lower, upper) bounds on allowable changes.
        """
        sensitivity_ranges = {}

        # For each constraint
        for r in range(self.solver.m):
            # Create unit vector e_r
            e_r = np.zeros(self.solver.m)
            e_r[r] = 1

            # Calculate A_B^(-1) * e_r
            A_B_inv_e_r = self.A_B_inv @ e_r

            # Calculate current basic variables' values
            current_basic_values = self.A_B_inv @ self.solver.b

            # Get current RHS value for this constraint
            current_rhs = self.solver.b[r]

            # Initialize allowable changes
            allowable_decrease = float('inf')
            allowable_increase = float('inf')

            # For each row of the inverted basis matrix
            for i in range(self.solver.m):
                # If coefficient is positive, it constrains how much we can decrease b_r
                if A_B_inv_e_r[i] > 0:
                    # How much we can decrease b_r without making x_B[i] negative
                    ratio = current_basic_values[i] / A_B_inv_e_r[i]
                    if ratio > 0:  # Only consider positive ratios
                        allowable_decrease = min(allowable_decrease, ratio)

                # If coefficient is negative, it constrains how much we can increase b_r
                elif A_B_inv_e_r[i] < 0:
                    # How much we can increase b_r without making x_B[i] negative
                    ratio = -current_basic_values[i] / A_B_inv_e_r[i]
                    if ratio > 0:  # Only consider positive ratios
                        allowable_increase = min(allowable_increase, ratio)

            # Handle infinite values
            if allowable_decrease == float('inf'):
                lower_bound = -np.inf
            else:
                lower_bound = current_rhs - allowable_decrease

            if allowable_increase == float('inf'):
                upper_bound = np.inf
            else:
                upper_bound = current_rhs + allowable_increase

            # Store the range for this constraint
            sensitivity_ranges[r] = (lower_bound, upper_bound)

        return sensitivity_ranges

    def objective_sensitivity_analysis(self):
        """
        Perform sensitivity analysis on the objective function coefficients.

        Returns:
        --------
        dict: Dictionary mapping variable indices to (lower, upper) bounds on allowable changes.
        """
        sensitivity_ranges = {}

        # Get the reduced costs from the tableau
        reduced_costs = self.solver.tableau[0, :-1]

        # For each variable
        for j in range(self.solver.n):
            if j in self.nonbasis_indices:  # Non-basic variable
                # The current reduced cost
                current_reduced_cost = reduced_costs[j]

                # Get current value of coefficient
                current_coef = self.solver.c[j]

                # Allowable increase - how much we can increase before basis changes
                allowable_increase = -current_reduced_cost

                # The actual upper bound is current value + allowable increase
                upper_bound = current_coef + allowable_increase

                # Lower bound remains the same
                lower_bound = -np.inf

                sensitivity_ranges[j] = (lower_bound, upper_bound)

            else:  # Basic variable
                # Get the corresponding row in the basis
                basis_position = self.basis_indices.index(j)

                # Extract the row from the tableau that corresponds to this basic variable
                row = self.solver.tableau[basis_position + 1, :-1]

                lower_bound = -np.inf
                upper_bound = np.inf

                # Check each non-basic variable
                for k in self.nonbasis_indices:
                    # Skip slack variables
                    if k >= self.solver.n:
                        continue

                    # If the entry is non-zero
                    if row[k] != 0:
                        # Calculate the ratio of reduced cost to tableau entry
                        ratio = reduced_costs[k] / row[k]

                        # Update bounds based on the sign of the tableau entry
                        if row[k] > 0:
                            upper_bound = min(upper_bound, ratio)
                        else:
                            lower_bound = max(lower_bound, ratio)

                sensitivity_ranges[j] = (lower_bound, upper_bound)

        return sensitivity_ranges

    def shadow_prices(self):
        """
        Calculate shadow prices (dual variables) for each constraint.

        Returns:
        --------
        numpy.ndarray: Array of shadow prices.
        """
        # Shadow prices are in the top row of the tableau for slack variables
        shadow_prices = np.zeros(self.solver.m)

        for i in range(self.solver.m):
            # The shadow price is the negative of the reduced cost of the slack variable
            slack_idx = self.solver.n + i
            shadow_prices[i] = -self.solver.tableau[0, slack_idx]

        return shadow_prices

    def format_range(self, range_tuple, var_value=None):
        """
        Format a sensitivity range in a readable way.

        Parameters:
        -----------
        range_tuple : tuple
            (lower_bound, upper_bound) tuple.
        var_value : float, optional
            Current value of the variable.

        Returns:
        --------
        str: Formatted range string.
        """
        lower, upper = range_tuple

        # Format lower bound
        if lower == -np.inf:
            lower_str = "-∞"
        else:
            lower_str = f"{lower:.4f}"

        # Format upper bound
        if upper == np.inf:
            upper_str = "+∞"
        else:
            upper_str = f"{upper:.4f}"

        # If var_value is provided, include the allowable delta
        if var_value is not None:
            delta_lower = "any decrease" if lower == -np.inf else f"{var_value - lower:.4f}"
            delta_upper = "any increase" if upper == np.inf else f"{upper - var_value:.4f}"

            return f"[{lower_str}, {upper_str}] (Current: {var_value:.4f}, Allowable Δ-: {delta_lower}, Allowable Δ+: {delta_upper})"
        else:
            return f"[{lower_str}, {upper_str}]"


def add_sensitivity_analysis_to_simplex(PrimalSimplex):
    """
    Extend the PrimalSimplex class with sensitivity analysis methods.
    """

    def perform_sensitivity_analysis(self):
        """
        Perform sensitivity analysis on the optimal solution.

        Returns:
        --------
        SensitivityAnalysis: Instance containing the sensitivity analysis results.
        """
        return SensitivityAnalysis(self)

    # Add the method to the class
    PrimalSimplex.perform_sensitivity_analysis = perform_sensitivity_analysis

    return PrimalSimplex


def solve_lp_scipy(c, A, b):
    """
    Solve a linear programming problem using SciPy's linprog function:
    Minimize c^T x
    Subject to A x >= b, x >= 0

    :param c: Coefficients of the objective function (1D array).
    :param A: Coefficient matrix of constraints (2D array).
    :param b: Right-hand side values of constraints (1D array).
    :return: Optimal solution (x) and optimal value (Z).
    """
    # Solve the LP problem
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

    # Check if the solution is successful
    if result.success:
        return result.x, result.fun
    else:
        raise ValueError("Problem could not be solved:", result.message)




# Example Usage
if __name__ == "__main__":
    # Example problem:
    # Minimize Z = 3x1 + 4x2
    # Subject to:
    # x1 + x2 >= 2
    # 2x1 + x2 >= 3
    # x1, x2 >= 0

    # Convert to standard form:
    # Minimize Z = 3x1 + 4x2
    # Subject to:
    # -x1 - x2 + s1 = -2
    # -2x1 - x2 + s2 = -3
    # x1, x2, s1, s2 >= 0

    problems = [

        {
            "c": np.array([2, 3]),
            "A": np.array([[1, 2], [2, 1], [1, 1]]),
            "b": np.array([10, 8, 5])
        },
        # Problem 2
        {
            "c": np.array([-1, -2]),
            "A": np.array([[1, 1], [2, 1]]),
            "b": np.array([3, 4])
        },
        # Problem 3
        {
            "c": np.array([-2, -3]),
            "A": np.array([[1, 2], [3, 1]]),
            "b": np.array([5, 7])
        },
        # Problem 4
        {
            "c": np.array([-4, -1]),
            "A": np.array([[1, 1], [1, -1]]),
            "b": np.array([2, 1])
        },
        # Problem 5
        {
            "c": np.array([-1, -1]),
            "A": np.array([[1, 0], [0, 1]]),
            "b": np.array([1, 1])
        },
        # Problem 6
        {
            "c": np.array([-5, -2]),
            "A": np.array([[2, 1], [1, 3]]),
            "b": np.array([6, 9])
        },
        # Problem 7
        {
            "c": np.array([-2, -4]),
            "A": np.array([[1, 1], [2, 2]]),
            "b": np.array([3, 6])
        },
        # Problem 8
        {
            "c": np.array([-3, -2]),
            "A": np.array([[1, 0], [0, 2]]),
            "b": np.array([2, 4])
        },
        # Problem 9
        {
            "c": np.array([-1, -3]),
            "A": np.array([[1, 1], [1, -1]]),
            "b": np.array([2, 1])
        },
        # Problem 10
        {
            "c": np.array([1, 1]),
            "A": np.array([[-1, 2], [-1, -2], [-1, -2]]),
            "b": np.array([-1, -4, 2])
        },
        {
            "c": np.array([-3, -2, -4, -1, -5]),
            "A": np.array([
                [2, 1, 3, 1, 2],  # Constraint 1
                [1, 2, 1, 3, 1],  # Constraint 2
                [3, 1, 2, 1, 4],  # Constraint 3
                [1, 1, 1, 1, 1],  # Constraint 4
            ]),
            "b": np.array([10, 8, 12, 6])
        },
        {
            "c": np.array([2, 3]),
            "A": np.array([[1, 2], [2, 1], [1, 1]]),
            "b": np.array([10, 8, 5])
        },
        {
            "c": np.array([-3, -2]),
            "A": np.array([[2, 1], [2, 3], [3, 1]]),
            "b": np.array([18, 42, 24])
        }
    ]

    for i, problem in enumerate(problems):
        print(f"Problem {i + 1}:")
        c, A, b = problem["c"], problem["A"], problem["b"]
        c_simplex = np.array(c, dtype=float)
        A_simplex = np.array(A, dtype=float)
        b_simplex = np.array(b, dtype=float)

        try:
            # Solve using Dual Simplex
            primal_simp = PrimalSimplex(c_simplex, A_simplex, b_simplex, use_fractions=False, fraction_digits=3,
                                        eq_constraints=False)
            solution_ds, optimal_value_ds = primal_simp.solve()
            print("Primal Simplex Solution:", solution_ds)
            print("Primal Simplex Optimal Value:", optimal_value_ds)
        except Exception as e:
            print("Dual Simplex Error:", str(e))

        try:
            # Solve using SciPy
            solution_sp, optimal_value_sp = solve_lp_scipy(c, A, b)
            print("SciPy Solution:", solution_sp)
            print("SciPy Optimal Value:", optimal_value_sp)
        except Exception as e:
            print("SciPy Error:", str(e))

        print("-" * 50)
