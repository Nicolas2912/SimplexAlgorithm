# test_simplex.py

import unittest
import numpy as np
from scipy.optimize import linprog
import sys
import os

# Add the directory containing simplex.py to the Python path
# Adjust this if your file structure is different
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the class to be tested
try:
    from simplex import PrimalSimplex
except ImportError as e:
    print(f"Failed to import PrimalSimplex from simplex.py: {e}")
    print("Ensure simplex.py is in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)


class TestPrimalSimplex(unittest.TestCase):

    def assertSolutionsAlmostEqual(self, simplex_res, scipy_res, places=6, atol=1e-6):
        """Helper method to compare simplex and scipy results."""
        simplex_x, simplex_z = simplex_res
        scipy_x, scipy_z = scipy_res

        # Compare objective values
        self.assertAlmostEqual(simplex_z, scipy_z, places=places,
                               msg=f"Objective values differ: Simplex={simplex_z}, SciPy={scipy_z}")

        # Compare solution vectors (consider permutations for basic/non-basic)
        # For standard problems where SciPy finds the same vertex, allclose works.
        # Be mindful if multiple optimal solutions exist, simplex might find a different vertex.
        self.assertTrue(np.allclose(simplex_x, scipy_x, atol=atol),
                        msg=f"Solution vectors differ:\nSimplex={simplex_x}\nSciPy  ={scipy_x}")

    def solve_with_scipy(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        """Helper to run scipy.linprog and handle results."""
        if bounds is None:
            bounds = [(0, None)] * len(c) # Default non-negativity

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs') #'highs' is generally robust

        if not result.success:
            # Allow certain statuses like infeasible or unbounded if expected by the test
            if result.status in [2, 3]: # 2: Infeasible, 3: Unbounded
                return None, result.status # Return status code instead of value
            else:
                 self.fail(f"SciPy linprog failed: {result.message} (Status: {result.status})")

        return result.x, result.fun

    # --- Test Cases ---

    def test_simple_le_problem_2d(self):
        """Test a standard 2D minimization problem with <= constraints."""
        # Minimize: z = -3x1 - 5x2  (From utils.create_example_2d)
        # Subject to:
        #   x1 <= 4
        #   2x2 <= 12
        #   3x1 + 2x2 <= 18
        #   x1, x2 >= 0
        # Optimal: x1=2, x2=6, z = -36
        c = np.array([-3.0, -5.0])
        A = np.array([
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 2.0]
        ])
        b = np.array([4.0, 12.0, 18.0])

        # Simplex Solver
        solver = PrimalSimplex(c, A, b, eq_constraints=False)
        simplex_x, simplex_obj_max_tableau = solver.solve()
        simplex_z = -simplex_obj_max_tableau # Convert back to Min value

        # SciPy Solver
        scipy_x, scipy_z = self.solve_with_scipy(c, A_ub=A, b_ub=b)

        self.assertSolutionsAlmostEqual((simplex_x, simplex_z), (scipy_x, scipy_z))
        # Check path vertices were recorded
        self.assertIsInstance(solver.path_vertices, list)
        self.assertGreater(len(solver.path_vertices), 0)
        self.assertEqual(solver.path_vertices[0].shape, (len(c),))

    def test_simple_le_problem_3d(self):
        """Test a standard 3D minimization problem with <= constraints."""
        # Minimize: -2x1 - 3x2 - 4x3 (From utils.create_example_3d)
        # Subject to:
        #   x1 + x2 + x3 <= 6
        #   2*x1 + x2 + 0*x3 <= 4
        #   0*x1 + x2 + 3*x3 <= 7
        #   x1, x2, x3 >= 0
        # Optimal solution should be around x = [0.5, 3, 4/3], z = -16.333...
        c = np.array([-2.0, -3.0, -4.0])
        A = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 3.0],
        ])
        b = np.array([6.0, 4.0, 7.0])

        # Simplex Solver
        solver = PrimalSimplex(c, A, b, eq_constraints=False)
        simplex_x, simplex_obj_max_tableau = solver.solve()
        simplex_z = -simplex_obj_max_tableau # Convert back to Min value

        # SciPy Solver
        scipy_x, scipy_z = self.solve_with_scipy(c, A_ub=A, b_ub=b)

        self.assertSolutionsAlmostEqual((simplex_x, simplex_z), (scipy_x, scipy_z))
        # Check basis identification
        basis_map = solver.get_basis_variable_indices()
        self.assertIsInstance(basis_map, dict)
        self.assertEqual(len(basis_map), A.shape[0]) # Should have m entries

    def test_equality_problem_phase1_2(self):
        """Test a problem requiring Phase I and Phase II (equality constraints)."""
        # Minimize z = 2x1 + 3x2
        # Subject to:
        #    x1 + x2 = 5
        #   2x1 - x2 = 1
        #   x1, x2 >= 0
        # Optimal: x1=2, x2=3, z = 4 + 9 = 13
        c = np.array([2.0, 3.0])
        A_eq = np.array([
            [1.0, 1.0],
            [2.0, -1.0]
        ])
        b_eq = np.array([5.0, 1.0])

        # Simplex Solver
        # Pass the equality constraints directly
        solver = PrimalSimplex(c, A_eq, b_eq, eq_constraints=True)
        simplex_x, simplex_obj_max_tableau = solver.solve()
        simplex_z = -simplex_obj_max_tableau # Convert back to Min value

        # SciPy Solver
        scipy_x, scipy_z = self.solve_with_scipy(c, A_eq=A_eq, b_eq=b_eq)

        self.assertSolutionsAlmostEqual((simplex_x, simplex_z), (scipy_x, scipy_z))

    def test_unbounded_problem(self):
        """Test detection of an unbounded problem."""
        # Minimize -x1 - x2
        # Subject to:
        # x1 - x2 <= 1
        # -x1 + x2 <= 1
        # x1, x2 >= 0
        # Objective can decrease indefinitely along x1=x2
        c = np.array([-1.0, -1.0])
        A = np.array([
            [1.0, -1.0],
            [-1.0, 1.0]
        ])
        b = np.array([1.0, 1.0])

        # Simplex Solver
        solver = PrimalSimplex(c, A, b, eq_constraints=False)
        # Expect an exception indicating unboundedness
        with self.assertRaisesRegex(Exception, "[Uu]nbounded"):
            solver.solve()

        # SciPy Solver (check status)
        scipy_x, scipy_status = self.solve_with_scipy(c, A_ub=A, b_ub=b)
        self.assertIsNone(scipy_x) # No optimal x for unbounded
        self.assertEqual(scipy_status, 3, msg="SciPy did not report unbounded status (3)")

    def test_infeasible_problem(self):
        """Test detection of an infeasible problem (using Phase I)."""
        # Minimize x1 + x2
        # Subject to:
        #   x1 + x2 = 1
        #   x1 + x2 = 3  (Contradictory)
        #   x1, x2 >= 0
        c = np.array([1.0, 1.0])
        A_eq = np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        b_eq = np.array([1.0, 3.0])

        # Simplex Solver
        solver = PrimalSimplex(c, A_eq, b_eq, eq_constraints=True)
        # Expect an exception indicating infeasibility from Phase I
        with self.assertRaisesRegex(Exception, "[Ii]nfeasible"):
            solver.solve()

        # SciPy Solver (check status)
        scipy_x, scipy_status = self.solve_with_scipy(c, A_eq=A_eq, b_eq=b_eq)
        self.assertIsNone(scipy_x) # No optimal x for infeasible
        self.assertEqual(scipy_status, 2, msg="SciPy did not report infeasible status (2)")

    def test_fraction_mode_execution(self):
        """Test if the solver runs without errors in fraction mode."""
        # Use the simple 2D problem again
        c = np.array([-3.0, -5.0])
        A = np.array([
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 2.0]
        ])
        b = np.array([4.0, 12.0, 18.0])

        try:
            # Simplex Solver in Fraction Mode
            solver = PrimalSimplex(c, A, b, use_fractions=True, fraction_digits=5, eq_constraints=False)
            simplex_x, simplex_obj_max_tableau = solver.solve()
            # No need to compare precisely, just ensure it finishes
            self.assertIsNotNone(simplex_x)
            self.assertIsNotNone(simplex_obj_max_tableau)
        except Exception as e:
            self.fail(f"Simplex solver crashed in fraction mode: {e}")
    
    def test_floating_point_problem(self):
        """Test a problem with floating-point coefficients and RHS."""
        # Minimize: z = 1.5x1 + 2.1x2 + 0.8x3
        # Subject to:
        #   0.5x1 + 1.2x2 + 0.3x3 <= 10.5
        #   2.0x1 + 0.8x2 + 1.5x3 <= 25.2
        #   1.0x1 + 1.0x2 + 1.0x3 <= 15.0
        #   x1, x2, x3 >= 0
        c = np.array([1.5, 2.1, 0.8])
        A = np.array([
            [0.5, 1.2, 0.3],
            [2.0, 0.8, 1.5],
            [1.0, 1.0, 1.0]
        ])
        b = np.array([10.5, 25.2, 15.0])

        # Simplex Solver
        solver = PrimalSimplex(c, A, b, eq_constraints=False)
        simplex_x, simplex_z = solver.solve()

        # SciPy Solver
        scipy_x, scipy_z = self.solve_with_scipy(c, A_ub=A, b_ub=b)

        # Use a slightly looser tolerance if necessary due to floating point arithmetic
        self.assertSolutionsAlmostEqual((simplex_x, simplex_z), (scipy_x, scipy_z), places=5, atol=1e-5)
        # Check path vertices
        self.assertIsInstance(solver.path_vertices, list)
        self.assertGreater(len(solver.path_vertices), 0)

# --- Run the tests ---
if __name__ == '__main__':
    unittest.main(verbosity=2)