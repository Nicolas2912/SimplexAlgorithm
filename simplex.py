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

    def build_initial_tableau_phase_I(self):
        """Build initial tableau for Phase I with proper handling of artificial variables"""
        num_slack = sum(1 for ct in self.constraint_types if ct == '<=')
        num_artificial = sum(1 for ct in self.constraint_types if ct == '=')
        self.n = self.n_vars + num_slack + num_artificial
        slack_count = 0
        artificial_count = 0
        self.tableau = []
        self.basis = []
        self.artificial_vars = []

        for i in range(self.m):
            row = [Fraction(0)] * (self.n + 1)
            for j in range(self.n_vars):
                row[j] = self.A[i][j]
            if self.constraint_types[i] == '<=':
                row[self.n_vars + slack_count] = Fraction(1)
                self.basis.append(self.n_vars + slack_count)
                slack_count += 1
            elif self.constraint_types[i] == '=':
                art_var = self.n_vars + num_slack + artificial_count
                row[art_var] = Fraction(1)
                self.artificial_vars.append(art_var)
                self.basis.append(art_var)
                artificial_count += 1
            row[-1] = self.b[i]
            self.tableau.append(row)

        # Phase I objective: minimize sum of artificial variables
        phase_I_obj = [Fraction(0)] * (self.n + 1)
        for art_var in self.artificial_vars:
            phase_I_obj[art_var] = Fraction(-1)
        self.tableau.append(phase_I_obj)

    def build_initial_tableau_phase_II(self):
        """
        Build Phase II tableau from Phase I solution with improved handling of artificial variables.
        """
        # First check if Phase I objective value is zero (feasible solution)
        if abs(self.tableau[-1][-1]) > self.tolerance:
            raise InfeasibleProblemError("Phase I did not reach zero objective value")

        # For each artificial variable in the basis
        artificial_in_basis = [(i, var) for i, var in enumerate(self.basis) if var in self.artificial_vars]
        
        for row_idx, art_var in artificial_in_basis:
            if abs(self.tableau[row_idx][-1]) > self.tolerance:
                # If artificial variable has non-zero value, try to pivot it out
                pivot_found = False
                
                # Look for a non-artificial variable to pivot in
                for j in range(self.n_vars):  # Only look at original variables
                    if j not in self.basis and abs(self.tableau[row_idx][j]) > self.tolerance:
                        # Found a potential pivot element
                        pivot_value = self.tableau[row_idx][j]
                        
                        # Check if this pivot maintains feasibility
                        feasible = True
                        for i in range(self.m):
                            if i != row_idx and self.tableau[i][j] != 0:
                                new_value = self.tableau[i][-1] - (self.tableau[i][j] * self.tableau[row_idx][-1] / pivot_value)
                                if new_value < -self.tolerance:
                                    feasible = False
                                    break
                        
                        if feasible:
                            # Perform the pivot
                            self.tableau[row_idx] = [elem / pivot_value for elem in self.tableau[row_idx]]
                            for i in range(self.m + 1):
                                if i != row_idx:
                                    factor = self.tableau[i][j]
                                    self.tableau[i] = [self.tableau[i][k] - factor * self.tableau[row_idx][k] 
                                                    for k in range(len(self.tableau[i]))]
                            self.basis[row_idx] = j
                            pivot_found = True
                            break
                            
                if not pivot_found:
                    raise InfeasibleProblemError("Unable to remove artificial variable from basis")

        # Remove artificial variables from the tableau
        indices_to_keep = [j for j in range(self.n + 1) if j not in self.artificial_vars]
        new_tableau = []
        for row in self.tableau[:-1]:  # Exclude the objective row
            new_row = [row[j] for j in indices_to_keep]
            new_tableau.append(new_row)
        
        # Update tableau dimensions
        self.n = len(indices_to_keep) - 1  # Subtract 1 for RHS column
        self.tableau = new_tableau

        # Build new objective row for Phase II
        objective = [Fraction(0)] * (self.n + 1)
        for j in range(self.n_vars):
            objective[j] = self.c[j] if self.problem_type == 'min' else -self.c[j]
            
        # Express objective in terms of basic variables
        for i in range(self.m):
            if self.basis[i] < self.n_vars:  # Only for original variables in basis
                coef = self.c[self.basis[i]] if self.problem_type == 'min' else -self.c[self.basis[i]]
                objective[-1] += coef * self.tableau[i][-1]
                for j in range(self.n):
                    objective[j] -= coef * self.tableau[i][j]
                    
        self.tableau.append(objective)

    def is_optimal_phase_I(self):
        """Check if Phase I solution is optimal"""
        # Check reduced costs
        reduced_costs = self.tableau[-1][:-1]
        if not all(rc <= self.tolerance for rc in reduced_costs):
            return False
            
        # Check if all artificial variables are zero
        for i, basic_var in enumerate(self.basis):
            if basic_var in self.artificial_vars and abs(self.tableau[i][-1]) > self.tolerance:
                return False
        
        return True

    def is_optimal_phase_II(self):
        """Check if Phase II solution is optimal"""
        reduced_costs = self.tableau[-1][:-1]
        if self.problem_type == 'min':
            return all(rc >= -self.tolerance for rc in reduced_costs)
        else:  # max
            return all(rc <= self.tolerance for rc in reduced_costs)
        
    def primal_simplex_step(self, phase):
        """
        Perform one step of the primal simplex method using Bland's rule to prevent cycling.
        
        Parameters:
        -----------
        phase : int
            1 for Phase I, 2 for Phase II
            
        Returns:
        --------
        bool
            True if a pivot was performed, False if the solution is optimal
        
        Raises:
        -------
        UnboundedProblemError
            If the problem is determined to be unbounded
        """
        # Check if current solution is optimal
        if phase == 1 and self.is_optimal_phase_I():
            return False
        if phase == 2 and self.is_optimal_phase_II():
            return False
        
        # Bland's Rule Step 1: Choose entering variable
        # Select smallest index j where reduced cost is favorable
        entering_var = -1
        for j in range(len(self.tableau[-1]) - 1):  # Exclude RHS
            reduced_cost = self.tableau[-1][j]
            if phase == 1:
                if reduced_cost > self.tolerance:
                    entering_var = j
                    break
            else:  # phase == 2
                if ((self.problem_type == 'max' and reduced_cost > self.tolerance) or 
                    (self.problem_type == 'min' and reduced_cost < -self.tolerance)):
                    entering_var = j
                    break
                    
        if entering_var == -1:
            return False  # No entering variable found - optimal solution reached
            
        # Bland's Rule Step 2: Choose leaving variable
        # Among all candidates that pass minimum ratio test, select smallest index
        min_ratio = float('inf')
        leaving_candidates = []  # Store all candidates that pass minimum ratio test
        
        for i in range(self.m):
            if self.tableau[i][entering_var] > self.tolerance:
                ratio = self.tableau[i][-1] / self.tableau[i][entering_var]
                if abs(ratio - min_ratio) < self.tolerance:
                    leaving_candidates.append((i, self.basis[i]))
                elif ratio < min_ratio:
                    min_ratio = ratio
                    leaving_candidates = [(i, self.basis[i])]
                    
        if not leaving_candidates:
            raise UnboundedProblemError()
            
        # Select the candidate with the smallest basic variable index
        leaving_row = min(leaving_candidates, key=lambda x: x[1])[0]
        
        # Perform pivot operation
        pivot_value = self.tableau[leaving_row][entering_var]
        self.tableau[leaving_row] = [elem / pivot_value for elem in self.tableau[leaving_row]]
        
        for i in range(len(self.tableau)):
            if i != leaving_row:
                factor = self.tableau[i][entering_var]
                self.tableau[i] = [
                    self.tableau[i][j] - factor * self.tableau[leaving_row][j]
                    for j in range(len(self.tableau[i]))
                ]
                
        self.basis[leaving_row] = entering_var
        return True

    def solve(self):
        """Solve using two-phase simplex method"""
        # Phase I
        self.build_initial_tableau_phase_I()
        iteration = 0
        while iteration < self.iteration_limit:
            if self.show_iterations:
                print(f"\nIteration {iteration + 1} (Phase I):")
                self.display_table()
            if not self.primal_simplex_step(phase=1):
                break
            iteration += 1

        # Check if Phase I solution is feasible
        if abs(self.tableau[-1][-1]) > self.tolerance:
            raise InfeasibleProblemError("Phase I did not reach zero objective value")

        # Phase II
        self.build_initial_tableau_phase_II()
        iteration = 0
        while iteration < self.iteration_limit:
            if self.show_iterations:
                print(f"\nIteration {iteration + 1} (Phase II):")
                self.display_table()
            if not self.primal_simplex_step(phase=2):
                break
            iteration += 1

        # Get and display solution
        solution = self.get_solution()
        if self.show_final_results:
            self.display_final_results(solution)
        return solution

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
    c = [2, 1]
    A = [
        [1, -1],
        [1, 1],
    ]
    b = [1, 2]
    n_vars = 2
    constraint_types = ['<=', '=']
    
    # Example with all output
    print("\nExample with all output:")
    simplex = Simplex(c, A, b, n_vars, show_iterations=True, show_final_results=True, display_type='fraction', 
                      problem_type="min", constraint_types=constraint_types)
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

    # verify(c, A, b)