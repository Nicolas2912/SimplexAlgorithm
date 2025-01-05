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
                    rhs_negative = True# Use negative slack for transformed constraints
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
            # For equality constraints
            # We only need artificial variables (no slack variables)
            artificial_vars = np.eye(self.m)
            tableau = np.hstack((self.A, artificial_vars))

            # Store indices of artificial variables for Phase I
            self.artificial_indices = list(range(self.n, self.n + self.m))

            # Create auxiliary objective row for Phase I (sum of artificial variables)
            auxiliary_obj = np.zeros(self.n + self.m)
            auxiliary_obj[self.artificial_indices] = 1

            # For each row with artificial variable, subtract it from auxiliary objective
            for i in range(self.m):
                auxiliary_obj -= tableau[i, :]

            # Create Phase I tableau
            b_extended = self.b.reshape(-1, 1)
            tableau = np.vstack((auxiliary_obj, tableau))
            tableau = np.hstack((tableau, np.vstack((-np.sum(self.b), b_extended))))

            # Store original objective coefficients for Phase II
            self.original_obj = np.zeros(self.n + self.m)
            self.original_obj[:self.n] = self.c

            return tableau

    def _find_pivot_column(self):
        """
        Find the entering variable (pivot column) by choosing the most negative coefficient in the objective row.
        """
        obj_row = self.tableau[0, :-1]
        pivot_col = np.argmin(obj_row)
        return pivot_col if obj_row[pivot_col] < 0 else None

    def _find_pivot_row(self, pivot_col):
        """
        Find the leaving variable (pivot row) using the minimum ratio test.
        """
        ratios = []
        for i in range(1, self.m + 1):
            if self.tableau[i, pivot_col] > 0:
                ratios.append(self.tableau[i, -1] / self.tableau[i, pivot_col])
            else:
                ratios.append(np.inf)
        pivot_row = np.argmin(ratios) + 1
        return pivot_row if ratios[pivot_row - 1] != np.inf else None

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
        Execute Phase I of the two-phase simplex method for equality constraints.
        """
        self.in_phase_one = True  # Flag to indicate we're in Phase I
        iteration = 0
        while True:
            pivot_col = self._find_pivot_column()
            if pivot_col is None:
                break

            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row is None:
                self.in_phase_one = False
                return False  # Problem is infeasible

            self._pivot(pivot_row, pivot_col)
            iteration += 1

            if iteration > 1000:  # Prevent infinite loops
                self.in_phase_one = False
                raise Exception("Phase I exceeded maximum iterations")

        # Check if the auxiliary objective value is approximately zero
        if abs(self.tableau[0, -1]) > 1e-10:
            self.in_phase_one = False
            return False  # Problem is infeasible

        # Remove artificial variables from basis if they remain
        # First, store the original tableau dimensions
        rows, cols = self.tableau.shape

        # Create new tableau without artificial columns
        new_tableau = np.zeros((rows, self.n + 1))
        new_tableau[:, :self.n] = self.tableau[:, :self.n]
        new_tableau[:, -1] = self.tableau[:, -1]

        # Replace objective row with original objective
        new_tableau[0, :self.n] = self.original_obj[:self.n]

        # Update objective row based on current basic variables
        for j in range(self.n):
            col = self.tableau[:, j]
            if np.sum(col == 1) == 1 and np.sum(col == 0) == self.m:
                basic_row = np.where(col == 1)[0][0]
                if basic_row > 0:  # Skip the objective row
                    new_tableau[0, :] -= self.original_obj[j] * new_tableau[basic_row, :]

        self.tableau = new_tableau
        self.in_phase_one = False  # Phase I is complete
        return True

    def _print_tableau(self, iteration):
        """
        Print the simplex tableau with consistent column headers and reduced costs.
        """
        print(f"\nIteration {iteration}:")

        # Prepare headers based on the phase and constraint type
        var_headers = [f"x{i + 1}" for i in range(self.n)]

        if not self.eq_constraints:
            # For inequality constraints, include slack variables
            slack_headers = [f"s{i + 1}" for i in range(self.m)]
            headers = var_headers + slack_headers + ["RHS"]
        else:
            if hasattr(self, 'in_phase_one') and self.in_phase_one:
                # For Phase I of equality constraints, include artificial variables
                art_headers = [f"a{i + 1}" for i in range(self.m)]
                headers = var_headers + art_headers + ["RHS"]
            else:
                # For Phase II of equality constraints, only original variables
                headers = var_headers + ["RHS"]

        rows = []

        # Convert tableau entries to fractions if use_fractions is True
        if self.use_fractions:
            tableau_frac = np.vectorize(lambda x: self._limit_fraction(Fraction(x)))(self.tableau)
            rows.append(["z"] + [str(frac) for frac in tableau_frac[0, :]])
            # Add reduced costs row (same as z-row for non-basic variables)
            rows.append(["RC"] + [str(frac) for frac in tableau_frac[0, :-1]] + [""])
            for i in range(1, self.m + 1):
                rows.append([f"R{i}"] + [str(frac) for frac in tableau_frac[i, :]])
        else:
            rows.append(["z"] + list(self.tableau[0, :]))
            # Add reduced costs row (same as z-row for non-basic variables)
            # rows.append(["RC"] + list(self.tableau[0, :-1]) + [""])
            for i in range(1, self.m + 1):
                rows.append([f"R{i}"] + list(self.tableau[i, :]))

        print("Tableau:")
        print(tabulate(rows, headers=headers, floatfmt=".4f"))

        # Print the current basis
        basis = []
        num_cols = self.tableau.shape[1] - 1  # Exclude RHS column
        for i in range(num_cols):
            col = self.tableau[:, i]
            if np.sum(col == 1) == 1 and np.sum(col == 0) == self.m:
                if i < self.n:
                    basis.append(f"x{i + 1}")
                elif not self.eq_constraints or (hasattr(self, 'in_phase_one') and self.in_phase_one):
                    if not self.eq_constraints:
                        basis.append(f"s{i + 1 - self.n}")
                    else:
                        basis.append(f"a{i + 1 - self.n}")
        print("Current Basis:", basis)

    def _get_basic_variables(self):
        """
        Correctly identify basic variables from the tableau.
        """
        basic_vars = []
        used_rows = set()

        # For each column (excluding RHS)
        for j in range(self.tableau.shape[1] - 1):
            # Get column with z-row! IMPORTANT!
            col = self.tableau[:, j]

            # Check if this is a unit vector
            one_position = None
            is_unit = True

            for i, val in enumerate(col):
                if abs(val - 1.0) < 1e-10:  # Found a 1
                    if one_position is None:
                        one_position = i
                    else:
                        is_unit = False  # More than one 1 found
                        break
                elif abs(val) > 1e-10:  # Found non-zero that's not 1
                    is_unit = False
                    break

            # one_position without + 1 IMPORTANT!
            if is_unit and one_position is not None and (one_position) not in used_rows:
                basic_vars.append((j, one_position))
                used_rows.add(one_position)

        basic_vars = sorted(basic_vars, key=lambda x: x[1])
        return basic_vars

    def solve(self):
        """
        Solve the linear programming problem using either standard simplex (≤ constraints)
        or two-phase simplex method (= constraints)
        """
        iteration = 0

        if self.eq_constraints:
            # Execute Phase I for equality constraints
            print("\nStarting Phase I")
            self._print_tableau(0)  # Print initial tableau
            feasible = self._phase_one()
            if not feasible:
                raise Exception("Problem is infeasible")
            print("\nPhase I completed - Starting Phase II")
            iteration = 0  # Reset iteration counter for Phase II

        # Phase II (same for both types of constraints)
        self._print_tableau(iteration)

        while True:
            pivot_col = self._find_pivot_column()
            if pivot_col is None:
                break  # Optimal solution found

            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row is None:
                break

            iteration += 1
            print(f"\nPivot: Entering Column = x{pivot_col + 1}, Leaving Row = R{pivot_row}")
            self._pivot(pivot_row, pivot_col)
            self._print_tableau(iteration)

        # Extract the solution
        solution = np.zeros(self.n)
        basic_vars = self._get_basic_variables()

        # Correctly assign the RHS values to the solution array based on the column index
        for var_idx, row_idx in basic_vars:
            if var_idx < self.n:  # Only original variables, not slack
                solution[var_idx] = self.tableau[row_idx, -1]

        if self.rhs_negative:
            optimal_value = self.tableau[0, -1]
        else:
            optimal_value = -self.tableau[0, -1]
        return solution, optimal_value

class DualSimplex:
    def __init__(self, c, A, b):
        """
        Initialize the dual simplex method for the problem:
        Minimize c^T x
        Subject to Ax <= b, x >= 0

        :param c: numpy array, coefficients of the objective function
        :param A: numpy matrix, coefficients of the constraints
        :param b: numpy array, right-hand side of the constraints
        """
        self.c = c
        self.A = A
        self.b = b
        self.m, self.n = A.shape
        self.tableau = self._initialize_tableau()

    def _initialize_tableau(self):
        """
        Initialize the simplex tableau by adding slack variables.
        """
        # Add slack variables to convert inequalities to equalities
        slack_vars = np.eye(self.m)
        tableau = np.hstack((self.A, slack_vars))
        c_extended = np.hstack((self.c, np.zeros(self.m)))
        b_extended = self.b.reshape(-1, 1)
        tableau = np.vstack((c_extended, tableau))
        tableau = np.hstack((tableau, np.vstack((0, b_extended))))
        return tableau

    def _find_pivot_row(self):
        """
        Find the leaving variable (pivot row) by choosing the most negative RHS value.
        """
        rhs = self.tableau[1:, -1]
        pivot_row = np.argmin(rhs) + 1
        return pivot_row if rhs[pivot_row - 1] < 0 else None

    def _find_pivot_column(self, pivot_row):
        """
        Find the entering variable (pivot column) using the dual ratio test.
        """
        ratios = []
        for j in range(self.n + self.m):
            if self.tableau[pivot_row, j] < 0:
                ratios.append(abs(self.tableau[0, j] / self.tableau[pivot_row, j]))
            else:
                ratios.append(np.inf)
        pivot_col = np.argmin(ratios)
        return pivot_col if ratios[pivot_col] != np.inf else None

    def _pivot(self, pivot_row, pivot_col):
        """
        Perform the pivot operation to update the tableau.
        """
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_element
        for i in range(self.m + 1):
            if i != pivot_row:
                self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]

    def _two_phase_method(self):
        """
        Implement the two-phase method to ensure dual feasibility.
        """
        # Phase I: Solve the auxiliary problem to find a feasible starting point
        # Create the auxiliary problem by adding artificial variables
        artificial_vars = np.eye(self.m)
        aux_A = np.hstack((self.A, artificial_vars))
        aux_c = np.hstack((np.zeros(self.n), np.ones(self.m)))
        aux_b = self.b

        # Initialize the auxiliary tableau
        aux_slack_vars = np.eye(self.m)
        aux_tableau = np.hstack((aux_A, aux_slack_vars))
        aux_c_extended = np.hstack((aux_c, np.zeros(self.m)))
        aux_b_extended = aux_b.reshape(-1, 1)
        aux_tableau = np.vstack((aux_c_extended, aux_tableau))
        aux_tableau = np.hstack((aux_tableau, np.vstack((0, aux_b_extended))))

        # Solve the auxiliary problem using the dual simplex method
        aux_simplex = DualSimplex(aux_c, aux_A, aux_b)
        aux_simplex.tableau = aux_tableau
        aux_solution, aux_value = aux_simplex.solve()

        if aux_value != 0:
            raise Exception("Problem is infeasible")

        # Phase II: Use the feasible starting point to solve the original problem
        # Remove artificial variables and update the tableau
        self.tableau = self._initialize_tableau()

    def solve(self):
        """
        Solve the linear programming problem using the dual simplex method with two-phase method.
        """
        # Check if the initial tableau is dual feasible
        if not np.all(self.tableau[0, :-1] >= 0):
            print("Initial tableau is not dual feasible. Using two-phase method.")
            self._two_phase_method()

        while True:
            pivot_row = self._find_pivot_row()
            if pivot_row is None:
                break  # Primal feasible solution found

            pivot_col = self._find_pivot_column(pivot_row)
            if pivot_col is None:
                raise Exception("Problem is infeasible")

            self._pivot(pivot_row, pivot_col)

        # Extract the solution
        solution = np.zeros(self.n)
        for i in range(self.n):
            col = self.tableau[:, i]
            if np.sum(col == 1) == 1 and np.sum(col == 0) == self.m:
                solution[i] = self.tableau[np.where(col == 1)[0][0], -1]

        optimal_value = -self.tableau[0, -1]
        return solution, optimal_value

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
            primal_simp = PrimalSimplex(c_simplex, A_simplex, b_simplex, use_fractions=False, fraction_digits=3, eq_constraints=False)
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