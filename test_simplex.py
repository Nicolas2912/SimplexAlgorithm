import unittest
import numpy as np
from fractions import Fraction
from simplex import PrimalSimplex  # Assuming the code is in primal_simplex.py


class TestPrimalSimplex(unittest.TestCase):
    def setUp(self):
        """Set up common test data"""
        # Simple 2-variable LP problem:
        # Minimize: -2x1 - 3x2
        # Subject to:
        #   x1 + 2x2 ≤ 4
        #   x1 + x2 ≤ 3
        #   x1, x2 ≥ 0
        self.simple_c = np.array([-2, -3])
        self.simple_A = np.array([[1, 2], [1, 1]])
        self.simple_b = np.array([4, 3])

    def test_initialization(self):
        """Test proper initialization of the simplex solver"""
        solver = PrimalSimplex(self.simple_c, self.simple_A, self.simple_b)

        self.assertEqual(solver.m, 2)  # Number of constraints
        self.assertEqual(solver.n, 2)  # Number of variables
        self.assertEqual(solver.tableau.shape, (3, 5))  # (m+1) × (n+m+1)

    def test_simple_lp_solution(self):
        """Test solving a simple LP problem with known solution"""
        solver = PrimalSimplex(self.simple_c, self.simple_A, self.simple_b)
        solution, optimal_value = solver.solve()

        # Expected solution: x1 = 1, x2 = 1.5
        # Expected optimal value: -6.5
        np.testing.assert_array_almost_equal(solution, [2, 1], decimal=4)
        self.assertAlmostEqual(optimal_value, -7.0, places=4)

    def test_unbounded_problem(self):
        """Test detection of unbounded problems"""
        # Minimize: -x1 - x2
        # Subject to:
        #   x1 - x2 ≤ 1
        #   x1, x2 ≥ 0
        c = np.array([-1, -1])
        A = np.array([[1, -1]])
        b = np.array([1])

        solver = PrimalSimplex(c, A, b)
        with self.assertRaises(Exception) as context:
            solver.solve()
        self.assertTrue("Problem is unbounded" in str(context.exception))

    def test_negative_rhs_handling(self):
        """Test proper handling of negative RHS values"""
        # Problem with negative RHS
        c = np.array([-1, -1])
        A = np.array([[1, 1], [-1, -1]])
        b = np.array([2, -1])  # Second constraint has negative RHS

        solver = PrimalSimplex(c, A, b)
        solution, _ = solver.solve()

        # Check if solution satisfies original constraints
        self.assertTrue(np.all(np.dot(A, solution) <= b))

    def test_fraction_output(self):
        """Test fraction output functionality"""
        solver = PrimalSimplex(
            self.simple_c,
            self.simple_A,
            self.simple_b,
            use_fractions=True,
            fraction_digits=3
        )

        # Test fraction limitation
        test_value = 22 / 7  # A common fraction that needs limitation
        limited_fraction = solver._limit_fraction(test_value)
        self.assertIsInstance(limited_fraction, Fraction)
        self.assertTrue(abs(float(limited_fraction) - test_value) < 0.1)

    def test_degenerate_case(self):
        """Test handling of degenerate cases"""
        # Degenerate problem where multiple constraints intersect at a point
        c = np.array([-1, -1])
        A = np.array([[1, 1], [1, 1], [1, 0]])
        b = np.array([2, 2, 1])

        solver = PrimalSimplex(c, A, b)
        solution, optimal_value = solver.solve()

        # Check if solution satisfies all constraints
        self.assertTrue(np.all(np.dot(A, solution) <= b + 1e-10))

    def test_zero_objective_coefficients(self):
        """Test handling of zero coefficients in objective function"""
        c = np.array([0, -1])
        A = np.array([[1, 1]])
        b = np.array([1])

        solver = PrimalSimplex(c, A, b)
        solution, optimal_value = solver.solve()

        # Check if solution is feasible
        self.assertTrue(np.all(np.dot(A, solution) <= b + 1e-10))

    def test_single_variable(self):
        """Test problem with single variable"""
        c = np.array([-1])
        A = np.array([[1]])
        b = np.array([2])

        solver = PrimalSimplex(c, A, b)
        solution, optimal_value = solver.solve()

        self.assertAlmostEqual(solution[0], 2, places=4)
        self.assertAlmostEqual(optimal_value, -2, places=4)

    def test_numerical_stability(self):
        """Test numerical stability with ill-conditioned matrices"""
        # Create an ill-conditioned matrix
        c = np.array([-1, -1])
        A = np.array([[1, 1.001], [1.001, 1]])
        b = np.array([2, 2])

        solver = PrimalSimplex(c, A, b, eq_constraints=True)
        solution, _ = solver.solve()

        # Check if solution remains feasible despite numerical challenges
        self.assertTrue(np.all(np.abs(np.dot(A, solution) - b) < 1e-6))

    def test_cycling_prevention(self):
        """Test prevention of cycling in degenerate cases"""
        # A known cycling example from literature
        c = np.array([-2, -3, -1, 0])
        A = np.array([
            [2, 1, 1, 0],
            [1, 2, 0, 1]
        ])
        b = np.array([4, 4])

        solver = PrimalSimplex(c, A, b, eq_constraints=True)
        solution, _ = solver.solve()

        # Check if solution is feasible
        self.assertTrue(np.all(np.abs(np.dot(A, solution) - b) < 1e-8))

    def test_small_coefficients(self):
        """Test handling of very small coefficients"""
        c = np.array([-1e-6, -2e-6])
        A = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        b = np.array([5e-6, 6e-6])

        # Test the equality constraints case
        solver_eq = PrimalSimplex(c, A, b, eq_constraints=True)
        with self.assertRaises(Exception) as context:
            solver_eq.solve()
        self.assertTrue("Problem is infeasible" in str(context.exception))

if __name__ == '__main__':
    unittest.main()