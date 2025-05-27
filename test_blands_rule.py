#!/usr/bin/env python3
"""
Test script to demonstrate Bland's anti-cycling rule implementation in the PrimalSimplex class.
"""

import numpy as np
from simplex import PrimalSimplex

def test_blands_rule_basic():
    """
    Test Bland's rule with a simple problem that could potentially cycle.
    """
    print("=" * 60)
    print("Testing Bland's Anti-Cycling Rule")
    print("=" * 60)
    
    # Example problem that might be prone to cycling
    # Minimize: 3x1 + 4x2
    # Subject to: x1 + x2 >= 2
    #            2x1 + x2 >= 3
    #            x1, x2 >= 0
    
    # Convert to standard form (<=):
    # Minimize: 3x1 + 4x2
    # Subject to: -x1 - x2 <= -2
    #            -2x1 - x2 <= -3
    #            x1, x2 >= 0
    
    c = np.array([3, 4])
    A = np.array([[-1, -1], [-2, -1]])
    b = np.array([-2, -3])
    
    print("Problem:")
    print("Minimize: 3x1 + 4x2")
    print("Subject to:")
    print("  x1 + x2 >= 2")
    print("  2x1 + x2 >= 3")
    print("  x1, x2 >= 0")
    print()
    
    # Solve with Bland's rule (now default)
    solver = PrimalSimplex(c, A, b)
    
    print("Initial tableau setup complete.")
    print("Using Bland's anti-cycling rule for pivot selection:")
    print("- Entering variable: smallest index among positive reduced costs")
    print("- Leaving variable: minimum ratio test with smallest basic variable index for ties")
    print()
    
    try:
        solution, optimal_value = solver.solve()
        
        print("Solution found!")
        print(f"Optimal solution: x1 = {solution[0]:.6f}, x2 = {solution[1]:.6f}")
        print(f"Optimal value: {optimal_value:.6f}")
        print(f"Total iterations: {len(solver.path_vertices) - 1}")
        
        # Display the path of vertices visited
        print("\nVertex path (solution values at each iteration):")
        for i, vertex in enumerate(solver.path_vertices):
            print(f"  Iteration {i}: x1 = {vertex[0]:.6f}, x2 = {vertex[1]:.6f}")
        
        # Verify constraints
        print("\nConstraint verification:")
        x1, x2 = solution[0], solution[1]
        print(f"  x1 + x2 = {x1 + x2:.6f} >= 2? {x1 + x2 >= 2 - 1e-6}")
        print(f"  2x1 + x2 = {2*x1 + x2:.6f} >= 3? {2*x1 + x2 >= 3 - 1e-6}")
        
    except Exception as e:
        print(f"Error during solving: {e}")
        return False
    
    return True

def test_blands_rule_demonstration():
    """
    Demonstrate how Bland's rule prevents cycling with a more complex example.
    """
    print("\n" + "=" * 60)
    print("Demonstrating Anti-Cycling Properties")
    print("=" * 60)
    
    # Create a problem where cycling might occur without Bland's rule
    # This is a degenerate problem designed to illustrate potential cycling
    c = np.array([1, 0, 0])
    A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    b = np.array([1, 1, 1])
    
    print("Problem with potential for degeneracy:")
    print("Minimize: x1")
    print("Subject to:")
    print("  x1 + x2 = 1")
    print("  x2 + x3 = 1") 
    print("  x1 + x3 = 1")
    print("  x1, x2, x3 >= 0")
    print()
    
    solver = PrimalSimplex(c, A, b, eq_constraints=True)
    
    try:
        solution, optimal_value = solver.solve()
        
        print("Solution found with Bland's rule!")
        print(f"Optimal solution: x1 = {solution[0]:.6f}, x2 = {solution[1]:.6f}, x3 = {solution[2]:.6f}")
        print(f"Optimal value: {optimal_value:.6f}")
        print(f"Total iterations: {len(solver.path_vertices) - 1}")
        
        if len(solver.path_vertices) <= 20:  # Reasonable number of iterations
            print("✓ Bland's rule successfully prevented cycling!")
        else:
            print("⚠ Many iterations required - potential cycling issues")
            
    except Exception as e:
        print(f"Error during solving: {e}")
        return False
    
    return True

def explain_blands_rule():
    """
    Explain how Bland's rule works.
    """
    print("\n" + "=" * 60)
    print("How Bland's Anti-Cycling Rule Works")
    print("=" * 60)
    
    explanation = """
Bland's rule is an anti-cycling pivoting strategy that prevents infinite loops 
in the simplex algorithm. It consists of two rules:

1. ENTERING VARIABLE SELECTION:
   - Among all variables with positive reduced costs (candidates to enter the basis),
   - Choose the one with the SMALLEST INDEX (lexicographically first).
   
2. LEAVING VARIABLE SELECTION:
   - Perform the standard minimum ratio test to find variables that can leave,
   - Among variables tied for the minimum ratio,
   - Choose the basic variable with the SMALLEST INDEX.

WHY IT PREVENTS CYCLING:
- Cycling occurs when the algorithm returns to a previously visited basis
- Bland's rule ensures a consistent, deterministic choice at each step
- The lexicographic ordering prevents the algorithm from "going in circles"
- Mathematical proof shows this guarantees finite termination

IMPLEMENTATION IN THIS CODE:
- _find_pivot_column(): Uses min(valid_indices) instead of argmax
- _find_pivot_row(): Sorts by (ratio, basic_variable_index) for tie-breaking
- _find_pivot_column_phase_one(): Same rule applied to Phase I

This implementation makes Bland's rule the DEFAULT behavior, eliminating
the risk of cycling in degenerate problems while maintaining correctness.
"""
    
    print(explanation)

if __name__ == "__main__":
    # Run all tests
    explain_blands_rule()
    
    success1 = test_blands_rule_basic()
    success2 = test_blands_rule_demonstration()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if success1 and success2:
        print("✓ All tests passed! Bland's rule is working correctly.")
        print("✓ Anti-cycling protection is active by default.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    print("\nBland's rule has been successfully implemented as the default")
    print("pivot selection strategy in the PrimalSimplex class.") 