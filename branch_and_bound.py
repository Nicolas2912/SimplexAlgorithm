import heapq
from collections import namedtuple
import numpy as np  # Import numpy
from scipy.optimize import milp, LinearConstraint, Bounds # Import scipy components

# Define Item structure (same as before)
Item = namedtuple("Item", ["index", "value", "weight", "ratio"])

# Define Node structure for the search tree (same as before)
class Node:
    def __init__(self, level, value, weight, bound, items_taken):
        self.level = level
        self.value = value
        self.weight = weight
        self.bound = bound
        self.items_taken = items_taken

    def __lt__(self, other):
        return self.bound > other.bound

    def __repr__(self):
        return f"Node(L:{self.level}, V:{self.value}, W:{self.weight}, B:{self.bound:.2f})"

# --- Bounding Function (same as before) ---
def calculate_bound(node, capacity, items, n):
    if node.weight > capacity:
        return 0
    bound_value = node.value
    remaining_capacity = capacity - node.weight
    current_level = node.level + 1
    while current_level < n and remaining_capacity > 0:
        item = items[current_level]
        if item.weight <= remaining_capacity:
            remaining_capacity -= item.weight
            bound_value += item.value
        else:
            fraction = remaining_capacity / item.weight
            bound_value += item.value * fraction
            remaining_capacity = 0
        current_level += 1
    return bound_value

# --- Branch and Bound Algorithm (same as before) ---
def knapsack_branch_and_bound(capacity, items):
    n = len(items)
    items_copy = items[:] # Work with a copy to keep original order for verification
    items_copy.sort(key=lambda x: x.ratio, reverse=True)

    priority_queue = []
    initial_items_taken = [None] * n
    root = Node(level=-1, value=0, weight=0, bound=0.0, items_taken=initial_items_taken)
    root.bound = calculate_bound(root, capacity, items_copy, n)
    heapq.heappush(priority_queue, root)

    max_value = 0
    best_items_taken = initial_items_taken[:]
    nodes_visited = 0

    while priority_queue:
        nodes_visited += 1
        current_node = heapq.heappop(priority_queue)

        if current_node.bound <= max_value:
            continue

        next_level = current_node.level + 1
        if next_level == n:
            continue

        # Item from the *sorted* list
        item = items_copy[next_level]

        # Branch 1: Include item
        if current_node.weight + item.weight <= capacity:
            include_weight = current_node.weight + item.weight
            include_value = current_node.value + item.value
            include_items_taken = current_node.items_taken[:]
            include_items_taken[item.index] = 1 # Use original index

            if include_value > max_value:
                max_value = include_value
                best_items_taken = include_items_taken[:]

            include_node = Node(level=next_level, value=include_value, weight=include_weight, bound=0.0, items_taken=include_items_taken)
            include_node.bound = calculate_bound(include_node, capacity, items_copy, n)

            if include_node.bound > max_value:
                heapq.heappush(priority_queue, include_node)

        # Branch 2: Exclude item
        exclude_weight = current_node.weight
        exclude_value = current_node.value
        exclude_items_taken = current_node.items_taken[:]
        exclude_items_taken[item.index] = 0 # Use original index

        exclude_node = Node(level=next_level, value=exclude_value, weight=exclude_weight, bound=0.0, items_taken=exclude_items_taken)
        exclude_node.bound = calculate_bound(exclude_node, capacity, items_copy, n)

        if exclude_node.bound > max_value:
            heapq.heappush(priority_queue, exclude_node)

    print(f"Branch&Bound Nodes visited: {nodes_visited}")

    # Finalize selection based on the best path found
    final_selection = [0] * n
    for i in range(n):
        if best_items_taken[i] == 1:
            final_selection[i] = 1

    return max_value, final_selection


# --- SciPy Verification Function ---
def verify_with_scipy(capacity, values, weights):
    """
    Solves the 0/1 Knapsack problem using scipy.optimize.milp for verification.

    Args:
        capacity (int): The maximum weight capacity of the knapsack.
        values (list): List of item values.
        weights (list): List of item weights.

    Returns:
        tuple: (max_value, selection_list) or (None, None) if failed.
    """
    n = len(values)
    if n == 0:
        return 0, []

    c = -np.array(values)  # Objective function coefficients (negative for maximization)
    A = [weights]         # Constraint matrix (weights)
    b_u = [capacity]      # Upper bound for constraints (capacity)
    b_l = [-np.inf]       # Lower bound for constraints

    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones_like(c) # All variables are integer
    bounds = Bounds(0, 1)         # Bounds for each variable (0 to 1)

    print("Running SciPy MILP solver...")
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if result.success:
            max_value = round(-result.fun) # Objective is negative, round for safety
            selection = np.round(result.x).astype(int).tolist() # Round results to 0 or 1
            return int(max_value), selection
        else:
            print(f"SciPy MILP failed. Status: {result.status}, Message: {result.message}")
            return None, None
    except Exception as e:
        print(f"An error occurred during SciPy MILP execution: {e}")
        return None, None


# --- Example Usage and Verification ---
if __name__ == "__main__":
    examples = [
        {
            "name": "Example 1: Simple case",
            "capacity": 10,
            "values": [10, 10, 12, 18],
            "weights": [2, 4, 6, 9]
        },
        {
            "name": "Example 2: Wikipedia example",
            "capacity": 165,
            "values": [92, 57, 49, 68, 60, 43, 67, 84, 87, 72],
            "weights": [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
        },
        {
            "name": "Example 3: Edge Case (High Capacity)",
            "capacity": 500,
            "values": [60, 100, 120],
            "weights": [10, 20, 30]
         },
        {
            "name": "Example 4: Edge Case (Low Capacity)",
            "capacity": 5,
            "values": [60, 100, 120],
            "weights": [10, 20, 30]
        },
         {
            "name": "Example 5: Tie in Ratio",
            "capacity": 8,
            "values": [15, 15, 16, 10], # Items 0 & 1 have same ratio
            "weights": [3, 3, 4, 2]
         }
    ]

    for i, ex in enumerate(examples):
        print(f"\n--- {ex['name']} ---")
        capacity = ex['capacity']
        values = ex['values']
        weights = ex['weights']
        n_items = len(values)

        print(f"Capacity: {capacity}")
        print(f"Items (value, weight): {list(zip(values, weights))}")

        # Prepare items for Branch and Bound
        items = [Item(index=j, value=v, weight=w, ratio=v/w if w > 0 else float('inf'))
                 for j, (v, w) in enumerate(zip(values, weights))]

        # Run Branch and Bound
        print("\nRunning Branch and Bound...")
        bnb_max_val, bnb_selection = knapsack_branch_and_bound(capacity, items)
        print(f"Branch&Bound Max value: {bnb_max_val}")
        print(f"Branch&Bound Selected items (0/1): {bnb_selection}")
        bnb_weight = sum(weights[j] for j, sel in enumerate(bnb_selection) if sel == 1)
        print(f"Branch&Bound Total weight: {bnb_weight}")
        bnb_selected_details = [(values[j], weights[j]) for j, sel in enumerate(bnb_selection) if sel == 1]
        print(f"Branch&Bound Selected item details: {bnb_selected_details}")


        # Run SciPy Verification
        print("\nRunning SciPy Verification...")
        scipy_max_val, scipy_selection = verify_with_scipy(capacity, values, weights)

        if scipy_max_val is not None:
            print(f"SciPy Max value: {scipy_max_val}")
            print(f"SciPy Selected items (0/1): {scipy_selection}")
            scipy_weight = sum(weights[j] for j, sel in enumerate(scipy_selection) if sel == 1)
            print(f"SciPy Total weight: {scipy_weight}")
            scipy_selected_details = [(values[j], weights[j]) for j, sel in enumerate(scipy_selection) if sel == 1]
            print(f"SciPy Selected item details: {scipy_selected_details}")


            # Comparison
            print("\nComparison:")
            value_match = (bnb_max_val == scipy_max_val)
            # Selection might differ if multiple optimal solutions exist, but value should match
            selection_match = (bnb_selection == scipy_selection)
            print(f"Max Value Match: {value_match}")
            if not value_match:
                 print(f"  WARNING: Maximum values differ! (BnB: {bnb_max_val}, SciPy: {scipy_max_val})")
            elif not selection_match:
                 print(f"  INFO: Selections differ, but max value matches. Both solutions are optimal.")
                 print(f"    BnB Selection : {bnb_selection}")
                 print(f"    SciPy Selection: {scipy_selection}")
            else:
                 print("  Selection Match: True")
        else:
            print("\nSciPy verification failed to produce a result.")
        print("-" * (len(ex['name']) + 6))