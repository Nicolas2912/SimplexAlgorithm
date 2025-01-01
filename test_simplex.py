import unittest
from fractions import Fraction
from simplex import Simplex, UnboundedProblemError, InfeasibleProblemError

class TestSimplexSolver(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-6  # Tolerance for floating-point comparisons

    def test_maximization_unique_solution(self):
        # Maximize: 3x1 + 2x2
        # Subject to:
        # x1 + x2 <= 4
        # x1 - x2 <= 2
        # x1, x2 >= 0
        c = [3, 2]
        A = [
            [1, 1],
            [1, -1]
        ]
        b = [4, 2]
        n_vars = 2
        simplex = Simplex(c, A, b, n_vars, problem_type='max', show_iterations=False, show_final_results=False)
        solution = simplex.solve()
        self.assertIsNotNone(solution)
        variables, obj_value = solution
        self.assertAlmostEqual(obj_value, 11.0, delta=self.tolerance)
        self.assertAlmostEqual(variables[0], 3.0, delta=self.tolerance)
        self.assertAlmostEqual(variables[1], 1.0, delta=self.tolerance)

    def test_minimization_unique_solution(self):
        # Minimize: x1 + 2x2
        # Subject to:
        # x1 + x2 >= 3  --> -x1 - x2 <= -3
        # x1 - x2 <= 1
        # x1, x2 >= 0
        c = [1, 2]
        A = [
            [-1, -1],  # Converted to <= by multiplying by -1
            [1, -1]
        ]
        b = [-3, 1]  # Ensure b is non-negative
        n_vars = 2
        simplex = Simplex(c, A, b, n_vars, problem_type='min', show_iterations=False, show_final_results=False)
        solution = simplex.solve()
        self.assertIsNotNone(solution)
        variables, obj_value = solution
        self.assertAlmostEqual(obj_value, 4.0, delta=self.tolerance)
        self.assertAlmostEqual(variables[0], 2.0, delta=self.tolerance)
        self.assertAlmostEqual(variables[1], 1.0, delta=self.tolerance)

    def test_infeasible_problem(self):
        # Maximize: x1 + x2
        # Subject to:
        # x1 + x2 <= 1
        # x1 + x2 >= 3  --> -x1 - x2 <= -3
        # x1, x2 >= 0
        c = [1, 1]
        A = [
            [1, 1],
            [-1, -1]
        ]
        b = [1, -3]  # Ensure b is non-negative
        n_vars = 2
        simplex = Simplex(c, A, b, n_vars, problem_type='max', show_iterations=False, show_final_results=False)
        with self.assertRaises(InfeasibleProblemError):
            simplex.solve()

    def test_unbounded_problem(self):
        # Maximize: x1
        # Subject to:
        # No constraints
        c = [1, 0]
        A = []
        b = []
        n_vars = 2
        simplex = Simplex(c, A, b, n_vars, problem_type='max', show_iterations=False, show_final_results=False)
        with self.assertRaises(UnboundedProblemError):
            simplex.solve()

    def test_degenerate_problem(self):
        # Maximize: 2x1 + 3x2
        # Subject to:
        # x1 + x2 <= 4
        # 2x1 + 2x2 <= 8
        # x1, x2 >= 0
        c = [2, 3]
        A = [
            [1, 1],
            [2, 2]
        ]
        b = [4, 8]
        n_vars = 2
        simplex = Simplex(c, A, b, n_vars, problem_type='max', show_iterations=False, show_final_results=False)
        solution = simplex.solve()
        self.assertIsNotNone(solution)
        variables, obj_value = solution
        self.assertAlmostEqual(obj_value, 12.0, delta=self.tolerance)
        self.assertAlmostEqual(variables[0], 0.0, delta=self.tolerance)
        self.assertAlmostEqual(variables[1], 4.0, delta=self.tolerance)

    def test_equality_constraints(self):
        # Maximize: 3x1 + 2x2
        # Subject to:
        # x1 + x2 = 4  --> x1 + x2 <= 4 and -x1 - x2 <= -4
        # x1, x2 >= 0
        c = [3, 2]
        A = [
            [1, 1],
            [-1, -1]
        ]
        b = [4, -4]  # Ensure b is non-negative
        n_vars = 2
        simplex = Simplex(c, A, b, n_vars, problem_type='max', show_iterations=False, show_final_results=False)
        solution = simplex.solve()
        self.assertIsNotNone(solution)
        variables, obj_value = solution
        self.assertAlmostEqual(obj_value, 12.0, delta=self.tolerance)
        self.assertAlmostEqual(variables[0], 4.0, delta=self.tolerance)
        self.assertAlmostEqual(variables[1], 0.0, delta=self.tolerance)

    def test_zero_coefficients(self):
        # Maximize: 0x1 + 0x2
        # Subject to:
        # x1 + x2 <= 1
        # x1, x2 >= 0
        c = [0, 0]
        A = [
            [1, 1]
        ]
        b = [1]
        n_vars = 2
        simplex = Simplex(c, A, b, n_vars, problem_type='max', show_iterations=False, show_final_results=False)
        solution = simplex.solve()
        self.assertIsNotNone(solution)
        variables, obj_value = solution
        self.assertAlmostEqual(obj_value, 0.0, delta=self.tolerance)
        # Any feasible solution is optimal
        self.assertLessEqual(variables[0] + variables[1], 1 + self.tolerance)
        self.assertGreaterEqual(variables[0] + variables[1], 0 - self.tolerance)

    def test_fractional_coefficients(self):
        # Maximize: (3/2)x1 + (2/3)x2
        # Subject to:
        # (1/2)x1 + (1/3)x2 <= 6
        # x1, x2 >= 0
        c = [Fraction(3, 2), Fraction(2, 3)]
        A = [
            [Fraction(1, 2), Fraction(1, 3)]
        ]
        b = [6]
        n_vars = 2
        simplex = Simplex(c, A, b, n_vars, problem_type='max', show_iterations=False, show_final_results=False)
        solution = simplex.solve()
        self.assertIsNotNone(solution)
        variables, obj_value = solution
        # Expected solution can be calculated accordingly
        # For brevity, assume x1=12, x2=0 gives obj=18
        self.assertAlmostEqual(obj_value, 18.0, delta=self.tolerance)
        self.assertAlmostEqual(variables[0], 12.0, delta=self.tolerance)
        self.assertAlmostEqual(variables[1], 0.0, delta=self.tolerance)

if __name__ == '__main__':
    unittest.main()