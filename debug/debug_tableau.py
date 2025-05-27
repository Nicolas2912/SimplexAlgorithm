import numpy as np
import sys
import os

# Add the directory containing simplex.py to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from simplex import PrimalSimplex

# Test the exact problem from test_edge_cases
c = np.array([1.0])
A = np.array([[1.0]])
b = np.array([5.0])

print('Problem setup:')
print(f'Minimize: {c[0]}*x1')
print(f'Subject to: {A[0,0]}*x1 <= {b[0]}')
print(f'x1 >= 0')
print()

# Our simplex solver
solver = PrimalSimplex(c, A, b, eq_constraints=False)

print("Initial tableau:")
print(solver.tableau)
print()

print("Basic variables info:")
basic_vars = solver._get_basic_variables()
print(f"Basic variables: {basic_vars}")
print()

# Debug the optimization direction
print("Debugging objective function direction:")
print(f"Original c (for minimization): {solver.c}")
print(f"Tableau row 0 (objective row): {solver.tableau[0, :]}")
print("Note: For minimization, we maximize -c, so tableau should have -c coefficients")
print()

# Now solve
simplex_x, simplex_obj_max_tableau = solver.solve()
simplex_z = -simplex_obj_max_tableau  # Convert back to Min value

print('Final tableau:')
print(solver.tableau)
print()

print('Final basic variables info:')
final_basic_vars = solver._get_basic_variables()
print(f"Final basic variables: {final_basic_vars}")
print()

print('Solution extraction:')
solution = np.zeros(solver.n)
for col_idx, row_idx in final_basic_vars:
    if col_idx < solver.n:
        val = solver.tableau[row_idx, -1]
        print(f"Variable x{col_idx+1} (col {col_idx}) is basic in row {row_idx} with value {val}")
        solution[col_idx] = max(0.0, val)

print(f"Extracted solution: {solution}")
print(f"Returned solution: {simplex_x}")
print(f"Objective value: {simplex_z}")
print()

print("Analysis:")
print("If x1 = 5, then objective = 1*5 = 5")
print("If x1 = 0, then objective = 1*0 = 0")
print("For minimization, x1 = 0 should be optimal, not x1 = 5") 