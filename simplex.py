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
        if problem_type.lower() == 'max':
                    c = [-ci for ci in c]

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
        # Count variables needed
        num_slack = sum(1 for ct in self.constraint_types if ct == '<=')
        num_surplus = sum(1 for ct in self.constraint_types if ct == '>=')
        num_artificial = sum(1 for ct in self.constraint_types if ct in ['=', '>='])
        
        # Total number of variables (original + slack + surplus + artificial)
        self.n = self.n_vars + num_slack + num_surplus + num_artificial
        
        # Initialize counters and lists
        slack_count = 0
        surplus_count = 0
        artificial_count = 0
        self.tableau = []
        self.basis = []
        self.artificial_vars = []

        # Build each constraint row
        for i in range(self.m):
            row = [Fraction(0)] * (self.n + 1)
            
            # Copy original coefficients
            for j in range(self.n_vars):
                row[j] = self.A[i][j]
            
            current_pos = self.n_vars
            
            if self.constraint_types[i] == '<=':
                # Add slack variable
                row[current_pos + slack_count] = Fraction(1)
                self.basis.append(current_pos + slack_count)
                slack_count += 1
                
            elif self.constraint_types[i] == '>=':
                # Add surplus variable
                row[current_pos + num_slack + surplus_count] = Fraction(-1)
                # Add artificial variable
                art_var = current_pos + num_slack + num_surplus + artificial_count
                row[art_var] = Fraction(1)
                self.artificial_vars.append(art_var)
                self.basis.append(art_var)
                surplus_count += 1
                artificial_count += 1
                
            elif self.constraint_types[i] == '=':
                # Add artificial variable
                art_var = current_pos + num_slack + num_surplus + artificial_count
                row[art_var] = Fraction(1)
                self.artificial_vars.append(art_var)
                self.basis.append(art_var)
                artificial_count += 1
            
            row[-1] = self.b[i]
            self.tableau.append(row)

        # Create Phase I objective row (minimize sum of artificial variables)
        phase_I_obj = [Fraction(0)] * (self.n + 1)
        
        # For each artificial variable in basis, add its row to objective
        for i, basic_var in enumerate(self.basis):
            if basic_var in self.artificial_vars:
                for j in range(len(phase_I_obj)):
                    phase_I_obj[j] -= self.tableau[i][j]
                    
        self.tableau.append(phase_I_obj)

    def build_initial_tableau_phase_II(self):
        """
        Build initial tableau for Phase II.
        We always solve minimization problems internally.
        """
        # Remove artificial variables
        indices_to_keep = [j for j in range(self.n + 1) if j not in self.artificial_vars]
        new_tableau = []
        for row in self.tableau[:-1]:
            new_row = [row[j] for j in indices_to_keep]
            new_tableau.append(new_row)
        
        self.n = len(indices_to_keep) - 1
        self.tableau = new_tableau
        
        # Build objective row
        objective = [Fraction(0)] * (self.n + 1)
        
        # Set initial coefficients for original variables
        for j in range(self.n_vars):
            objective[j] = self.c[j]
        
        # Adjust objective row based on basic variables
        for i in range(self.m):
            if self.basis[i] < self.n_vars:
                coef = self.c[self.basis[i]]
                for j in range(self.n):
                    objective[j] += coef * self.tableau[i][j]
                objective[-1] += coef * self.tableau[i][-1]
        
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
        """
        Check if Phase II solution is optimal.
        For minimization problems: check if all reduced costs are ≥ 0
        """
        reduced_costs = self.tableau[-1][:-1]  # Exclude RHS
        return all(rc >= -self.tolerance for rc in reduced_costs)
        
    def primal_simplex_step(self, phase):
        """Perform one step of the primal simplex method"""
        reduced_costs = self.tableau[-1][:-1]
        
        if phase == 1:
            if self.is_optimal_phase_I():
                return False
            # Phase I: Select most positive reduced cost
            entering_var = max(range(len(reduced_costs)), 
                            key=lambda j: reduced_costs[j])
            if reduced_costs[entering_var] <= self.tolerance:
                return False
        else:  # Phase II
            if self.is_optimal_phase_II():
                return False
            
            # For minimization: select most negative reduced cost
            entering_var = min(range(len(reduced_costs)), 
                            key=lambda j: reduced_costs[j])
            if reduced_costs[entering_var] >= -self.tolerance:
                return False

        # Find leaving variable using minimum ratio test
        min_ratio = float('inf')
        leaving_var = -1
        
        for i in range(self.m):
            if self.tableau[i][entering_var] > self.tolerance:
                ratio = float(self.tableau[i][-1]) / float(self.tableau[i][entering_var])
                if ratio < min_ratio and ratio >= 0:
                    min_ratio = ratio
                    leaving_var = i
        
        if leaving_var == -1:
            raise UnboundedProblemError()
        
        self._pivot(leaving_var, entering_var)
        return True
    
    def _pivot(self, leaving_var, entering_var):
        """
        Perform pivot operation at the specified position.
        This is extracted as a separate method for clarity and reuse.
        """
        # Normalize the pivot row
        pivot_value = self.tableau[leaving_var][entering_var]
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
        """Extract solution from final tableau"""
        solution = [Fraction(0)] * self.n_vars
        
        for i in range(self.m):
            if self.basis[i] < self.n_vars:
                solution[self.basis[i]] = self.tableau[i][-1]
        
        # Calculate objective value using original coefficients
        obj_value = sum(self.c[j] * solution[j] for j in range(self.n_vars))
        
        if self.problem_type == "max":
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
        """Display the current tableau with proper z-row calculations"""
        def format_number(num):
            if self.display_type == 'decimal':
                return f"{float(num):.6f}"
            else:
                return (f"{num.numerator}" if num.denominator == 1 
                    else f"{num.numerator}/{num.denominator}")
        
        # Calculate z-row values
        z_row = [Fraction(0)] * (self.n + 1)
        for j in range(self.n + 1):
            for i in range(self.m):
                if self.basis[i] < self.n_vars:
                    coef = self.c[self.basis[i]]
                    z_row[j] += coef * self.tableau[i][j]
        
        # Calculate reduced costs
        reduced_costs = []
        for j in range(self.n):
            if j < self.n_vars:
                rc = -self.c[j] - self.tableau[-1][j]  # Using tableau's objective row
            else:
                rc = -self.tableau[-1][j]
            reduced_costs.append(rc)
        
        # Prepare table data
        table_data = []
        z_row_display = ["z"] + [format_number(z) for z in z_row]
        table_data.append(z_row_display)
        rc_row = ["z-c"] + [format_number(rc) for rc in reduced_costs] + [""]
        table_data.append(rc_row)
        c_row = ["c"] + [format_number(c) for c in self.c] + ["0"] * (self.n - self.n_vars) + [""]
        table_data.append(c_row)
        table_data.append(["---"] * (self.n + 2))
        
        # Add basic variable rows
        for i in range(self.m):
            basis_var = f"x{self.basis[i]+1}" if self.basis[i] < self.n_vars else f"s{self.basis[i]-self.n_vars+1}"
            row_data = [basis_var] + [format_number(x) for x in self.tableau[i]]
            table_data.append(row_data)
        
        # Create headers and print
        headers = ["Basis"] + [f"x{i+1}" for i in range(self.n_vars)] + \
                 [f"s{i+1}" for i in range(self.m)] + ["b"]
        print(tabulate(table_data, headers=headers, tablefmt="outline", stralign="center"))
        print(f"Current basis: {set(f'x{self.basis[i]+1}' if self.basis[i] < self.n_vars else f's{self.basis[i]-self.n_vars+1}' for i in range(self.m))}")


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
    c = [1, 1]
    A = [
        [1, 1],
        [1, 1],
    ]
    b = [3, -1]
    n_vars = 2
    constraint_types = ['>=', '<=']
    problem_type = "max"
    
    # Example with all output
    print("\nExample with all output:")
    simplex = Simplex(c, A, b, n_vars, show_iterations=True, show_final_results=True, display_type='decimal', 
                      problem_type=problem_type, constraint_types=constraint_types)
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