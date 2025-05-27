import numpy as np
from scipy.optimize import linprog
from simplex import PrimalSimplex


def solve_with_primal_simplex(c, A, b):
    """Solve using PrimalSimplex class"""
    solver = PrimalSimplex(c, A, b)
    solution, optimal_value = solver.solve()
    return solution, optimal_value


def solve_with_scipy(c, A, b):
    """Solve using SciPy's linprog function"""
    # Note: SciPy's linprog minimizes by default, so we use c directly
    result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

    if result.success:
        return result.x, result.fun
    else:
        raise ValueError(f"Problem could not be solved: {result.message}")


def main():
    # Define the problem
    c = np.array([-3, -2])  # Objective coefficients
    A = np.array([
        [2, 1],  # First constraint
        [2, 3],  # Second constraint
        [3, 1]  # Third constraint
    ])
    b = np.array([18, 42, 24])  # RHS values

    print("Linear Programming Problem:")
    print("Minimize: -3x₁ - 2x₂")
    print("Subject to:")
    print("2x₁ + x₂ ≤ 18")
    print("2x₁ + 3x₂ ≤ 42")
    print("3x₁ + x₂ ≤ 24")
    print("x₁, x₂ ≥ 0")
    print("\n" + "-" * 50 + "\n")

    # Solve using PrimalSimplex
    print("Solving with PrimalSimplex:")
    try:
        ps_solution, ps_value = solve_with_primal_simplex(c, A, b)
        print(f"Solution: x₁ = {ps_solution[0]:.6f}, x₂ = {ps_solution[1]:.6f}")
        print(f"Optimal value: {ps_value:.6f}")
    except Exception as e:
        print(f"Error with PrimalSimplex: {str(e)}")

    print("\n" + "-" * 50 + "\n")

    # Solve using SciPy's linprog
    print("Solving with SciPy's linprog:")
    try:
        scipy_solution, scipy_value = solve_with_scipy(c, A, b)
        print(f"Solution: x₁ = {scipy_solution[0]:.6f}, x₂ = {scipy_solution[1]:.6f}")
        print(f"Optimal value: {scipy_value:.6f}")
    except Exception as e:
        print(f"Error with SciPy: {str(e)}")

    print("\n" + "-" * 50 + "\n")

    # Compare results
    if 'ps_solution' in locals() and 'scipy_solution' in locals():
        solution_diff = np.linalg.norm(ps_solution - scipy_solution)
        value_diff = abs(ps_value - scipy_value)

        print("Comparison:")
        print(f"Solution difference (L2 norm): {solution_diff:.10f}")
        print(f"Optimal value difference: {value_diff:.10f}")

        if solution_diff < 1e-5 and value_diff < 1e-5:
            print("✅ Results match within tolerance!")
        else:
            print("❌ Results differ significantly!")


if __name__ == "__main__":
    main()