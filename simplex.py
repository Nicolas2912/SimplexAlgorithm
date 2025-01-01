import sys
from fractions import Fraction
from scipy.optimize import linprog
import time
from tabulate import tabulate
import numpy as np


class UnboundedProblemError(Exception):
    """Exception raised when the problem is unbounded."""
    def __init__(self, message="The problem is unbounded."):
        self.message = message
        super().__init__(self.message)

class InfeasibleProblemError(Exception):
    """Exception raised when the problem is infeasible."""
    def __init__(self, message="The problem is infeasible."):
        self.message = message
        super().__init__(self.message)

class Simplex:
    def __init__(self, c, A, b, n_vars, constraint_types, problem_type='max', display_type='fraction', max_denominator_digits=4, show_iterations=True, show_final_results=True):
        """
        Initialize the Simplex solver with the problem data and display options.
        
        Additional Parameters:
        --------------------
        constraint_types : list of str
            List of constraint types ('<=', '>=', '=')
        show_iterations : bool
            Whether to display the tableau for each iteration
        show_final_results : bool
            Whether to display the final results (solution and objective value)
        """
        self.c = [Fraction(str(ci)) for ci in c]
        self.A = [[Fraction(str(aij)) for aij in row] for row in A]
        self.b = [Fraction(str(bi)) for bi in b]
        self.n_vars = n_vars
        self.constraint_types = constraint_types
        self.problem_type = problem_type.lower()
        self.m = len(b)  # number of constraints
        self.n = self.n_vars + self.m  # total number of variables including slack/surplus/artificial
        self.tableau = []  # will store the simplex tableau
        self.basis = []  # will store the indices of basic variables
        self.display_type = display_type
        self.max_denominator_digits = max_denominator_digits
        self.iteration_limit = 100
        self.tolerance = Fraction(1, 10**10)
        self.show_iterations = show_iterations
        self.show_final_results = show_final_results

    def build_initial_tableau(self):
        """
        Build the initial tableau for the primal simplex method with proper handling of equality constraints.
        """
        self.tableau = []
        self.basis = []
        artificial_vars = []

        # First pass: Count variables needed
        num_slack = sum(1 for ct in self.constraint_types if ct == '<=')
        num_surplus = sum(1 for ct in self.constraint_types if ct == '>=')
        num_artificial = sum(1 for ct in self.constraint_types if ct in ['=', '>='])

        # Update total number of variables
        total_extra_vars = num_slack + num_surplus + num_artificial
        self.n = self.n_vars + total_extra_vars

        # Second pass: Build tableau
        slack_surplus_count = 0
        artificial_count = 0
        
        for i in range(self.m):
            row = []
            # Original variables coefficients
            row.extend(self.A[i])
            
            # Add space for all slack and surplus variables
            for j in range(num_slack + num_surplus):
                row.append(Fraction(0))
                
            # Add space for all artificial variables
            for j in range(num_artificial):
                row.append(Fraction(0))

            # Now set the specific coefficients based on constraint type
            if self.constraint_types[i] == '<=':
                # Set slack variable coefficient
                row[self.n_vars + slack_surplus_count] = Fraction(1)
                self.basis.append(self.n_vars + slack_surplus_count)
                slack_surplus_count += 1
            
            elif self.constraint_types[i] == '>=':
                # Set surplus variable coefficient
                row[self.n_vars + slack_surplus_count] = Fraction(-1)
                slack_surplus_count += 1
                # Set artificial variable coefficient
                art_pos = self.n_vars + num_slack + num_surplus + artificial_count
                row[art_pos] = Fraction(1)
                artificial_vars.append(art_pos)
                self.basis.append(art_pos)
                artificial_count += 1
            
            elif self.constraint_types[i] == '=':
                # Set artificial variable coefficient
                art_pos = self.n_vars + num_slack + num_surplus + artificial_count
                row[art_pos] = Fraction(1)
                artificial_vars.append(art_pos)
                self.basis.append(art_pos)
                artificial_count += 1

            # Add RHS
            row.append(self.b[i])
            self.tableau.append(row)

        # Build objective row
        obj_row = [-coef for coef in self.c]  # Negated coefficients for max problem
        obj_row.extend([Fraction(0)] * (num_slack + num_surplus))  # Zeros for slack/surplus
        obj_row.extend([Fraction(1000000)] * num_artificial)  # Big M for artificial
        obj_row.append(Fraction(0))  # RHS of objective row
        self.tableau.append(obj_row)

        # Verify tableau dimensions
        n_cols = len(self.tableau[0])
        for row in self.tableau:
            if len(row) != n_cols:
                raise ValueError(f"Inconsistent tableau dimensions. Expected {n_cols} columns, got {len(row)}")

        return artificial_vars


    def is_feasible(self):
        """
        Check if the initial basic solution is feasible.
        """
        for i in range(self.m):
            if self.tableau[i][-1] < -self.tolerance:  # Check if RHS is negative
                return False
        return True

    def is_optimal(self):
        """
        Check if the current solution is optimal.
        For maximization: 
        - All reduced costs should be ≥ 0
        """
        reduced_costs = self.tableau[-1][:-1]
        if self.problem_type == 'max':
            return all(rc >= -self.tolerance for rc in reduced_costs)
        else:  # min
            return all(rc <= self.tolerance for rc in reduced_costs)

    def primal_simplex_step(self):
        """
        Perform one step of the primal simplex method.
        1. Select entering variable (most negative reduced cost for maximization)
        2. Select leaving variable (minimum ratio test)
        3. Perform pivot operation
        """
        if self.is_optimal():
            return False

        # Find the entering variable (most negative reduced cost for maximization)
        reduced_costs = self.tableau[-1][:-1]
        entering_var = -1
        min_rc = -self.tolerance
        
        for j, rc in enumerate(reduced_costs):
            if rc < min_rc:
                min_rc = rc
                entering_var = j
        
        if entering_var == -1:
            return False

        # Find the leaving variable using minimum ratio test
        leaving_var = -1
        min_ratio = float('inf')
        
        for i in range(self.m):
            coef = self.tableau[i][entering_var]
            if coef > self.tolerance:  # Consider only positive coefficients
                ratio = self.tableau[i][-1] / coef
                if ratio < min_ratio:
                    min_ratio = ratio
                    leaving_var = i
        
        if leaving_var == -1:
            return False

        # Perform pivot operation
        pivot_value = self.tableau[leaving_var][entering_var]
        
        # Update the pivot row
        self.tableau[leaving_var] = [elem / pivot_value for elem in self.tableau[leaving_var]]
        
        # Update all other rows
        for i in range(len(self.tableau)):
            if i != leaving_var:
                factor = self.tableau[i][entering_var]
                self.tableau[i] = [
                    self.tableau[i][j] - factor * self.tableau[leaving_var][j]
                    for j in range(len(self.tableau[i]))
                ]
        
        # Update basis
        self.basis[leaving_var] = entering_var
        return True

    def solve(self):
        """
        Solve the linear programming problem using primal or dual simplex method.
        Uses dual simplex if initial solution is dual feasible but primal infeasible.
        
        Raises:
            InfeasibleProblemError: If the problem is infeasible.
            UnboundedProblemError: If the problem is unbounded.
        """
        self.build_initial_tableau()
        iteration_count = 0
        
        # Check if we need dual simplex (negative b values)
        using_dual = not self.is_feasible()
        
        while iteration_count < self.iteration_limit:
            if self.show_iterations:
                print(f"\nIteration {iteration_count + 1}:")
                self.display_table()
            
            if using_dual:
                step_success = self.dual_simplex_step()
                if not step_success:
                    if self.is_feasible():
                        # Switch to primal if we achieved feasibility
                        using_dual = False
                        continue
                    else:
                        if self.show_iterations or self.show_final_results:
                            print("\nProblem is infeasible")
                        raise InfeasibleProblemError()
            else:
                if not self.primal_simplex_step():
                    if self.is_optimal():
                        solution = self.get_solution()
                        if self.show_iterations:
                            print("\nOptimal solution found")
                        self.display_final_results(solution)
                        return solution
                    else:
                        if self.show_iterations or self.show_final_results:
                            print("\nProblem is unbounded")
                        raise UnboundedProblemError()
            
            iteration_count += 1
        
        if self.show_iterations or self.show_final_results:
            print(f"\nReached iteration limit of {self.iteration_limit}")
        return None

    def get_solution(self):
        """
        Extract the solution from the final tableau.
        Returns:
        - List of values for the original variables
        - Optimal objective value
        """
        solution = [Fraction(0) for _ in range(self.n_vars)]
        
        for i in range(self.m):
            var = self.basis[i]
            if var < self.n_vars:
                solution[var] = self.tableau[i][-1]
        
        obj_value = sum(self.c[i] * solution[i] for i in range(self.n_vars))
        if self.problem_type == 'min':
            obj_value = -obj_value
        
        return [float(x) for x in solution], float(obj_value)

    def dual_simplex_step(self):
        """
        Perform one step of the dual simplex method.
        1. Select leaving variable (most negative RHS)
        2. Select entering variable (minimum ratio test with negative coefficients)
        3. Perform pivot operation
        
        Returns:
            bool: True if a step was performed, False if optimal or infeasible
        """
        # Find leaving variable (most negative RHS)
        leaving_var = -1
        min_rhs = -self.tolerance
        
        for i in range(self.m):
            rhs = self.tableau[i][-1]
            if rhs < min_rhs:
                min_rhs = rhs
                leaving_var = i
        
        if leaving_var == -1:
            return False  # No negative RHS, primal feasible
        
        # Find entering variable using dual ratio test
        entering_var = -1
        min_ratio = float('inf')
        
        reduced_costs = self.tableau[-1][:-1]  # Get reduced costs row
        for j in range(len(reduced_costs)):
            coef = self.tableau[leaving_var][j]
            if coef < -self.tolerance:  # Look for negative coefficients
                ratio = -reduced_costs[j] / coef
                if ratio < min_ratio:
                    min_ratio = ratio
                    entering_var = j
        
        if entering_var == -1:
            return False  # Problem is infeasible
        
        # Perform pivot operation
        pivot_value = self.tableau[leaving_var][entering_var]
        
        # Update the pivot row
        self.tableau[leaving_var] = [elem / pivot_value for elem in self.tableau[leaving_var]]
        
        # Update all other rows
        for i in range(len(self.tableau)):
            if i != leaving_var:
                factor = self.tableau[i][entering_var]
                self.tableau[i] = [
                    self.tableau[i][j] - factor * self.tableau[leaving_var][j]
                    for j in range(len(self.tableau[i]))
                ]
        
        # Update basis
        self.basis[leaving_var] = entering_var
        return True

    def display_table(self):
        """
        Display the current tableau in a formatted table with z-values row.
        Respects the display_type setting ('fraction' or 'decimal').
        """
        def format_number(num):
            """Helper function to format numbers according to display_type"""
            if self.display_type == 'decimal':
                return f"{float(num):.6f}"
            else:  # fraction
                return (f"{num.numerator}" if num.denominator == 1 
                    else f"{num.numerator}/{num.denominator}")

        # Extract reduced costs (last row excluding RHS)
        reduced_costs = self.tableau[-1][:-1]
        
        # Calculate z values for each column
        z_values = []
        for j in range(len(self.tableau[0])-1):  # Exclude RHS column
            z_val = Fraction(0)
            for i in range(self.m):  # For each basic variable
                # Get the objective coefficient for the basic variable
                if self.basis[i] < self.n_vars:
                    c_i = self.c[self.basis[i]]
                else:
                    c_i = Fraction(0)  # Slack variables have zero objective coefficient
                # Multiply by the column coefficient and add to sum
                z_val += c_i * self.tableau[i][j]
            z_values.append(z_val)
        
        # Prepare the table data
        table_data = []
        for i, row in enumerate(self.tableau[:-1]):
            basis_var = f"x{self.basis[i]+1}" if self.basis[i] < self.n_vars else f"s{self.basis[i]-self.n_vars+1}"
            row_display = [basis_var] + [format_number(elem) for elem in row]
            table_data.append(row_display)
        
        # Calculate and add the current objective value for z row
        obj_value = Fraction(0)
        for i in range(self.m):
            if self.basis[i] < self.n_vars:  # Only consider original variables, not slack
                c_i = self.c[self.basis[i]]
                b_i = self.tableau[i][-1]
                obj_value += c_i * b_i

        # Add the z values row with objective value
        z_row = ["z"] + [format_number(z) for z in z_values] + [format_number(obj_value)]
        
        # Add the reduced costs row
        reduced_costs_row = ["z-c"] + [format_number(rc) for rc in reduced_costs] + [""]
        
        # Create c values row (original objective coefficients)
        c_row = ["c"] + [format_number(c) for c in self.c] + ["0"] * self.m + [""]

        # Create table rows in specific order
        header = ["Basis"] + [f"x{i+1}" for i in range(self.n_vars)] + [f"s{i+1}" for i in range(self.m)] + ["b"]
        
        # Combine all rows in desired order
        all_rows = [header, z_row, reduced_costs_row, c_row] + table_data
        
        # Insert a separator row after the c row
        separator_row = ["---"] * len(header)
        all_rows.insert(4, separator_row)
        
        # Print the table using tabulate
        print(tabulate(all_rows, headers="firstrow", tablefmt="outline", stralign="center"))
        
        # Print the current basis in a set
        basis_set = {f"x{self.basis[i]+1}" if self.basis[i] < self.n_vars else f"s{self.basis[i]-self.n_vars+1}" 
                    for i in range(self.m)}
        print(f"Current basis: {basis_set}")

    def display_final_results(self, solution):
        """
        Display the final results of the simplex method.
        
        Parameters:
        -----------
        solution : tuple
            Tuple containing (variables values, objective value)
        """
        if not self.show_final_results:
            return
            
        primal_solution, obj_value = solution
        if self.display_type == 'fraction':
            # Convert to fractions for display
            frac_solution = [Fraction(str(val)).limit_denominator(10**self.max_denominator_digits) for val in primal_solution]
            frac_obj = Fraction(str(obj_value)).limit_denominator(10**self.max_denominator_digits)
            
            print("\nFinal Solution:")
            print("Variables:", [f"x{i+1} = {f'{val.numerator}/{val.denominator}' if val.denominator != 1 else str(val.numerator)}" 
                               for i, val in enumerate(frac_solution)])
            print("Objective Value:", f"{frac_obj.numerator}/{frac_obj.denominator}" if frac_obj.denominator != 1 else str(frac_obj.numerator))
        else:
            print("\nFinal Solution:")
            print("Variables:", [f"x{i+1} = {val:.6f}" for i, val in enumerate(primal_solution)])
            print("Objective Value:", f"{obj_value:.6f}")

def main():
    # Test Case: Simple maximization problem
    print("Test Case: Simple maximization problem")
    c = [3, 2]
    A = [
        [2, 1],
        [1, 2],
        [1, 1]
    ]
    b = [4, 3, 10]
    n_vars = 2
    constraint_types = [">=", ">=", "<="]
    
    # Example with all output
    print("\nExample with all output:")
    simplex = Simplex(c, A, b, n_vars, show_iterations=True, show_final_results=True, display_type='fraction', 
                      problem_type="max", constraint_types=constraint_types)
    simplex.solve()

def verify(c, A, b):
    """
    Verify the solution using scipy.optimize.linprog.
    
    Parameters:
    c (list): Coefficients of the objective function.
    A (list of lists): Coefficients of the inequality constraints.
    b (list): Right-hand side values of the inequality constraints.
    """
    # Solve the problem using scipy's linprog
    result = linprog(c=c, A_eq=A, b_eq=b, method='highs')
    
    # Check if the problem has a feasible solution
    if result.success:
        # Extract the solution and objective value
        variables = [f"x{i+1} = {result.x[i]}" for i in range(len(result.x))]
        objective_value = -result.fun  # Negate because linprog minimizes
        
        # Print the final result
        print("Final Solution (scipy):")
        print(f"Variables: {variables}")
        print(f"Objective Value: {objective_value}")
    else:
        print("The problem is infeasible or unbounded. (scipy)")

if __name__ == "__main__":
    main()

    c = [-2, -1]
    A = [
        [-1, 1],
        [1, 1]
    ]
    b = [-2, 4]
    n_vars = 2

    verify(c, A, b)