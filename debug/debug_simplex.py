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
simplex_x, simplex_obj_max_tableau = solver.solve()
simplex_z = -simplex_obj_max_tableau  # Convert back to Min value

print('Our Simplex solution:')
print(f'x = {simplex_x}')
print(f'objective = {simplex_z}')
print()

print('Problem analysis:')
print('This is a minimization problem: min 1*x1')
print('Constraints: x1 <= 5, x1 >= 0')
print('The optimal solution should be at x1 = 0 (lower bound)')
print('Our solver seems to be finding x1 = 5 (upper bound)')
print('This suggests the solver is maximizing instead of minimizing!') 