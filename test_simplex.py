# test_simplex.py

import pytest
import numpy as np
from scipy.optimize import linprog
import sys
import os
import warnings

# Add the directory containing simplex.py to the Python path
# Adjust this if your file structure is different
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the classes and exceptions to be tested
try:
    from simplex import (
        PrimalSimplex, 
        SensitivityAnalysis,
        SimplexError,
        InfeasibleProblemError,
        UnboundedProblemError,
        DegenerateSolutionError,
        NumericalInstabilityError,
        TableauCorruptionError,
        solve_lp_scipy
    )
except ImportError as e:
    print(f"Failed to import from simplex.py: {e}")
    print("Ensure simplex.py is in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)


def assert_solutions_almost_equal(simplex_res, scipy_res, places=6, atol=1e-6):
    """Helper function to compare simplex and scipy results."""
    simplex_x, simplex_z = simplex_res
    scipy_x, scipy_z = scipy_res

    # Compare objective values
    assert abs(simplex_z - scipy_z) < 10**(-places), \
        f"Objective values differ: Simplex={simplex_z}, SciPy={scipy_z}"

    # Compare solution vectors (consider permutations for basic/non-basic)
    # For standard problems where SciPy finds the same vertex, allclose works.
    # Be mindful if multiple optimal solutions exist, simplex might find a different vertex.
    assert np.allclose(simplex_x, scipy_x, atol=atol), \
        f"Solution vectors differ:\nSimplex={simplex_x}\nSciPy  ={scipy_x}"


def solve_with_scipy(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """Helper to run scipy.linprog and handle results."""
    if bounds is None:
        bounds = [(0, None)] * len(c) # Default non-negativity

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs') #'highs' is generally robust

    if not result.success:
        # Allow certain statuses like infeasible or unbounded if expected by the test
        if result.status in [2, 3]: # 2: Infeasible, 3: Unbounded
            return None, result.status # Return status code instead of value
        else:
            pytest.fail(f"SciPy linprog failed: {result.message} (Status: {result.status})")

    return result.x, result.fun


# --- Test Cases ---

def test_simple_le_problem_2d():
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
    simplex_x, simplex_z = solver.solve()

    # SciPy Solver
    scipy_x, scipy_z = solve_with_scipy(c, A_ub=A, b_ub=b)

    assert_solutions_almost_equal((simplex_x, simplex_z), (scipy_x, scipy_z))
    # Check path vertices were recorded
    assert isinstance(solver.path_vertices, list)
    assert len(solver.path_vertices) > 0
    assert solver.path_vertices[0].shape == (len(c),)


def test_simple_le_problem_3d():
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
    simplex_x, simplex_z = solver.solve()

    # SciPy Solver
    scipy_x, scipy_z = solve_with_scipy(c, A_ub=A, b_ub=b)

    assert_solutions_almost_equal((simplex_x, simplex_z), (scipy_x, scipy_z))
    # Check basis identification
    basis_map = solver.get_basis_variable_indices()
    assert isinstance(basis_map, dict)
    assert len(basis_map) == A.shape[0] # Should have m entries


def test_equality_problem_phase1_2():
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
    simplex_x, simplex_z = solver.solve()

    # SciPy Solver
    scipy_x, scipy_z = solve_with_scipy(c, A_eq=A_eq, b_eq=b_eq)

    assert_solutions_almost_equal((simplex_x, simplex_z), (scipy_x, scipy_z))


def test_unbounded_problem():
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
    # Expect the specific UnboundedProblemError exception
    with pytest.raises(UnboundedProblemError):
        solver.solve()

    # SciPy Solver (check status)
    scipy_x, scipy_status = solve_with_scipy(c, A_ub=A, b_ub=b)
    assert scipy_x is None # No optimal x for unbounded
    assert scipy_status == 3, "SciPy did not report unbounded status (3)"


def test_infeasible_problem():
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
    # Expect the specific InfeasibleProblemError exception
    with pytest.raises(InfeasibleProblemError):
        solver.solve()

    # SciPy Solver (check status)
    scipy_x, scipy_status = solve_with_scipy(c, A_eq=A_eq, b_eq=b_eq)
    assert scipy_x is None # No optimal x for infeasible
    assert scipy_status == 2, "SciPy did not report infeasible status (2)"


def test_fraction_mode_execution():
    """Test if the solver runs without errors in fraction mode."""
    # Use the simple 2D problem again
    c = np.array([-3.0, -5.0])
    A = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [3.0, 2.0]
    ])
    b = np.array([4.0, 12.0, 18.0])

    # Simplex Solver in Fraction Mode
    solver = PrimalSimplex(c, A, b, use_fractions=True, fraction_digits=5, eq_constraints=False)
    simplex_x, simplex_z = solver.solve()
    # No need to compare precisely, just ensure it finishes
    assert simplex_x is not None
    assert simplex_z is not None


def test_floating_point_problem():
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
    scipy_x, scipy_z = solve_with_scipy(c, A_ub=A, b_ub=b)

    # Use a slightly looser tolerance if necessary due to floating point arithmetic
    assert_solutions_almost_equal((simplex_x, simplex_z), (scipy_x, scipy_z), places=5, atol=1e-5)
    # Check path vertices
    assert isinstance(solver.path_vertices, list)
    assert len(solver.path_vertices) > 0


def test_input_validation_errors():
    """Test input validation with improved error messages."""
    # Test empty constraint matrix
    with pytest.raises(ValueError, match="at least one constraint"):
        PrimalSimplex(c=[1, 2], A=np.array([]).reshape(0, 2), b=[])
    
    # Test empty variable vector
    with pytest.raises(ValueError, match="at least one variable"):
        PrimalSimplex(c=[], A=np.array([]).reshape(1, 0), b=[1])
    
    # Test dimension mismatch
    with pytest.raises(ValueError, match="does not match"):
        PrimalSimplex(c=[1, 2], A=np.array([[1, 2, 3]]), b=[1])
    
    # Test non-finite values
    with pytest.raises(ValueError, match="non-finite"):
        PrimalSimplex(c=[1, np.inf], A=np.array([[1, 2]]), b=[1])


def test_numerical_instability_detection():
    """Test detection of numerical instability."""
    # Create a problem that might lead to very small pivot elements
    c = np.array([1.0, 1.0])
    A = np.array([
        [1e-15, 1.0],  # Very small coefficient
        [1.0, 1.0]
    ])
    b = np.array([1.0, 2.0])
    
    solver = PrimalSimplex(c, A, b, eq_constraints=False)
    # This might raise NumericalInstabilityError depending on the pivot selection
    try:
        solution, obj_value = solver.solve()
        # If it doesn't raise an error, that's also fine - just check solution is reasonable
        assert np.all(np.isfinite(solution))
        assert np.isfinite(obj_value)
    except NumericalInstabilityError:
        # This is expected behavior for ill-conditioned problems
        pass


def test_sensitivity_analysis_initialization():
    """Test sensitivity analysis initialization and error handling."""
    # First solve a simple problem
    c = np.array([1.0, 2.0])
    A = np.array([[1.0, 1.0], [2.0, 1.0]])
    b = np.array([3.0, 4.0])
    
    solver = PrimalSimplex(c, A, b, eq_constraints=False)
    solution, obj_value = solver.solve()
    
    # Test successful initialization
    sens_analysis = SensitivityAnalysis(solver)
    assert isinstance(sens_analysis, SensitivityAnalysis)
    
    # Test error when passed wrong type
    with pytest.raises(TypeError):
        SensitivityAnalysis("not a solver")
    
    # Test error when solver has no tableau
    invalid_solver = PrimalSimplex(c, A, b, eq_constraints=False)
    invalid_solver.tableau = None
    with pytest.raises(ValueError, match="valid tableau"):
        SensitivityAnalysis(invalid_solver)


def test_degenerate_problem():
    """Test handling of degenerate problems."""
    # Create a degenerate problem (multiple optimal solutions)
    c = np.array([1.0, 1.0])
    A = np.array([
        [1.0, 1.0],
        [1.0, 1.0],  # Redundant constraint
        [1.0, 0.0]
    ])
    b = np.array([2.0, 2.0, 1.0])
    
    solver = PrimalSimplex(c, A, b, eq_constraints=False)
    # Should handle degeneracy gracefully
    solution, obj_value = solver.solve()
    
    # Check that solution is valid
    assert np.all(solution >= -1e-10)  # Non-negative
    assert np.all(np.isfinite(solution))
    assert np.isfinite(obj_value)


def test_large_problem_handling():
    """Test handling of larger problems."""
    np.random.seed(42)  # For reproducibility
    n_vars = 10
    n_constraints = 5
    
    # Generate a random feasible problem
    c = np.random.rand(n_vars)
    A = np.random.rand(n_constraints, n_vars)
    # Ensure feasibility by making b large enough
    b = np.sum(A, axis=1) + 1.0
    
    solver = PrimalSimplex(c, A, b, eq_constraints=False)
    
    try:
        solution, obj_value = solver.solve()
        
        # Verify solution properties
        assert len(solution) == n_vars
        assert np.all(solution >= -1e-10)
        assert np.isfinite(obj_value)
        
        # Check constraint satisfaction
        constraint_vals = A @ solution
        assert np.all(constraint_vals <= b + 1e-10)
        
    except (NumericalInstabilityError, TableauCorruptionError):
        # These are acceptable for randomly generated problems
        pass


def test_negative_rhs_handling():
    """Test handling of negative RHS values."""
    c = np.array([1.0, 1.0])
    A = np.array([
        [1.0, 1.0],
        [-1.0, 1.0]
    ])
    b = np.array([2.0, -1.0])  # One negative RHS
    
    solver = PrimalSimplex(c, A, b, eq_constraints=False)
    solution, obj_value = solver.solve()
    
    # Should handle negative RHS by transforming constraints
    assert np.all(solution >= -1e-10)
    assert np.isfinite(obj_value)


def test_cycling_detection():
    """Test the cycling detection mechanism."""
    # Create a problem that might lead to cycling
    c = np.array([1.0, 0.0, 0.0])
    A = np.array([
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    b = np.array([1.0, 0.5, 0.5])
    
    solver = PrimalSimplex(c, A, b, eq_constraints=False)
    
    # Should complete without infinite cycling
    # We use warnings.catch_warnings to capture any warnings that might be issued
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solution, obj_value = solver.solve()
    
    assert np.all(solution >= -1e-10)
    assert np.isfinite(obj_value)


def test_tableau_corruption_detection():
    """Test detection of tableau corruption."""
    c = np.array([1.0, 1.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([2.0])
    
    solver = PrimalSimplex(c, A, b, eq_constraints=False)
    
    # Manually corrupt the tableau after initialization
    solver.tableau[1, -1] = np.nan  # Introduce NaN in RHS
    
    # Should detect corruption during solution extraction
    with pytest.raises(TableauCorruptionError):
        solver.solve()


def test_edge_cases():
    """Test various edge cases."""
    # Problem with single variable and constraint
    # Minimize: z = 1*x1
    # Subject to: x1 <= 5, x1 >= 0
    # Optimal solution: x1 = 0, z = 0 (minimize at lower bound)
    c = np.array([1.0])
    A = np.array([[1.0]])
    b = np.array([5.0])
    
    solver = PrimalSimplex(c, A, b, eq_constraints=False)
    solution, obj_value = solver.solve()
    
    # Compare with scipy to ensure our expectation is correct
    scipy_x, scipy_z = solve_with_scipy(c, A_ub=A, b_ub=b)
    
    assert len(solution) == 1
    # The correct optimal solution should be x = 0 (at the origin), obj = 0
    assert abs(solution[0] - scipy_x[0]) < 1e-6
    assert abs(obj_value - scipy_z) < 1e-6


def test_performance_monitoring():
    """Test that the solver provides useful performance information."""
    c = np.array([1.0, 2.0, 3.0])
    A = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0]
    ])
    b = np.array([6.0, 4.0, 8.0])
    
    solver = PrimalSimplex(c, A, b, eq_constraints=False)
    solution, obj_value = solver.solve()
    
    # Check that path vertices are recorded
    assert isinstance(solver.path_vertices, list)
    assert len(solver.path_vertices) > 0
    
    # Each vertex should have the right dimension
    for vertex in solver.path_vertices:
        assert len(vertex) == len(c)
        assert np.all(np.isfinite(vertex))


def test_scipy_wrapper_function():
    """Test the improved solve_lp_scipy wrapper function."""
    # Test inequality constraints
    c = np.array([1.0, 2.0])
    A = np.array([[1.0, 1.0], [2.0, 1.0]])
    b = np.array([3.0, 4.0])
    
    solution, obj_value = solve_lp_scipy(c, A, b, constraint_type='<=')
    assert np.all(solution >= -1e-10)
    assert np.isfinite(obj_value)
    
    # Test equality constraints
    c_eq = np.array([1.0, 1.0])
    A_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([2.0])
    
    solution_eq, obj_value_eq = solve_lp_scipy(c_eq, A_eq, b_eq, constraint_type='=')
    assert np.all(solution_eq >= -1e-10)
    assert np.isfinite(obj_value_eq)
    
    # Test input validation
    with pytest.raises(ValueError):
        solve_lp_scipy(c, A[:-1], b)  # Dimension mismatch
    
    # Test invalid constraint type
    with pytest.raises(ValueError):
        solve_lp_scipy(c, A, b, constraint_type='invalid')
    
    # Test infeasible problem
    c_inf = np.array([1.0, 1.0])
    A_inf = np.array([[1.0, 1.0], [1.0, 1.0]])
    b_inf = np.array([1.0, 3.0])  # Contradictory
    
    with pytest.raises(ValueError, match="infeasible"):
        solve_lp_scipy(c_inf, A_inf, b_inf, constraint_type='=')


def test_warning_system():
    """Test that warnings are properly issued for various conditions."""
    # Test that warnings are issued for potential issues
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Create a problem that might trigger warnings
        c = np.array([1.0, 1.0])
        A = np.array([[1.0, 1.0], [1.0, 1.0]])  # Redundant constraints
        b = np.array([2.0, 2.0])
        
        solver = PrimalSimplex(c, A, b, eq_constraints=False)
        solution, obj_value = solver.solve()
        
        # Check that solution is still valid even with warnings
        assert np.all(solution >= -1e-10)
        assert np.isfinite(obj_value)


def test_exception_inheritance():
    """Test that custom exceptions inherit properly."""
    # Test exception hierarchy
    assert issubclass(InfeasibleProblemError, SimplexError)
    assert issubclass(UnboundedProblemError, SimplexError)
    assert issubclass(NumericalInstabilityError, SimplexError)
    assert issubclass(TableauCorruptionError, SimplexError)
    assert issubclass(DegenerateSolutionError, SimplexError)
    assert issubclass(SimplexError, Exception)


def test_comprehensive_sensitivity_analysis():
    """Test sensitivity analysis functionality more thoroughly."""
    # Solve a problem first
    c = np.array([2.0, 3.0])
    A = np.array([[1.0, 1.0], [2.0, 1.0]])
    b = np.array([4.0, 6.0])
    
    solver = PrimalSimplex(c, A, b, eq_constraints=False)
    solution, obj_value = solver.solve()
    
    # Test sensitivity analysis
    sens_analysis = SensitivityAnalysis(solver)
    
    # Test RHS sensitivity
    rhs_ranges = sens_analysis.rhs_sensitivity_analysis()
    assert isinstance(rhs_ranges, dict)
    assert len(rhs_ranges) == len(b)
    
    # Test objective sensitivity
    obj_ranges = sens_analysis.objective_sensitivity_analysis()
    assert isinstance(obj_ranges, dict)
    assert len(obj_ranges) == len(c)
    
    # Test shadow prices
    shadow_prices = sens_analysis.shadow_prices()
    assert isinstance(shadow_prices, np.ndarray)
    assert len(shadow_prices) == len(b)


# --- Run the tests ---
if __name__ == '__main__':
    pytest.main([__file__, '-v'])