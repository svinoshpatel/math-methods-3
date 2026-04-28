import numpy as np
from math import floor, ceil, isfinite
from scipy.optimize import linprog

EPS = 1e-9

def is_integer_value(v, eps=EPS):
    return abs(v - round(v)) <= eps

def is_integer_solution(x, integer_indices, eps=EPS):
    return all(is_integer_value(x[i], eps) for i in integer_indices)

def choose_branch_variable(x, integer_indices, eps=EPS):
    for i in integer_indices:
        if not is_integer_value(x[i], eps):
            return i
    return None

def solve_lp_relaxation(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                        bounds=None, maximize=True, method="highs"):
    c = np.array(c, dtype=float)
    if maximize:
        lp_c = -c
    else:
        lp_c = c

    res = linprog(
        c=lp_c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method
    )

    if not res.success:
        return {
            "success": False,
            "status": res.status,
            "message": res.message,
            "x": None,
            "objective": None
        }
    
    x = res.x
    obj = -res.fun if maximize else res.fun

    return {
        "success": True,
        "status": res.status,
        "message": res.message,
        "x": x,
        "objective": obj
    }

def branch_and_bound(
    c,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    bounds=None,
    integer_indices=None,
    binary_indices=None,
    maximize=True,
    method="highs",
    verbose=True
):
    c = np.array(c, dtype=float)
    n = len(c)

    if A_ub is not None:
        A_ub = np.array(A_ub, dtype=float)
        b_ub = np.array(b_ub, dtype=float)

    if A_eq is not None:
        A_eq = np.array(A_eq, dtype=float)
        b_eq = np.array(b_eq, dtype=float)

    if bounds is None:
        bounds = [(0, None)] * n
    else:
        bounds = list(bounds)

    if integer_indices is None:
        integer_indices = list(range(n))
    else:
        integer_indices = list(integer_indices)

    if binary_indices is None:
        binary_indices = []
    else:
        binary_indices = list(binary_indices)

    for j in binary_indices:
        if j not in integer_indices:
            integer_indices.append(j)

    integer_indices = sorted(set(integer_indices))

    bounds = list(bounds)
    for j in binary_indices:
        lb, ub = bounds[j]
        lb = 0 if lb is None else max(lb, 0)
        ub = 1 if ub is None else min(ub, 1)
        bounds[j] = (lb, ub)

    if maximize:
        best_obj = -np.inf
    else:
        best_obj = np.inf
    best_x = None

    stack = [{
        "bounds": bounds.copy(),
        "depth": 0,
        "name": "root"
    }]

    visited_nodes = 0

    while stack:
        node = stack.pop()
        visited_nodes += 1

        node_bounds = node["bounds"]

        if verbose:
            print("\n" + "=" * 70)
            print(f"Node: {node['name']} | depth={node['depth']}")
            print(f"Bounds: {node_bounds}")
            print("=" * 70)

        result = solve_lp_relaxation(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=node_bounds,
            maximize=maximize,
            method=method
        )

        if not result["success"]:
            if verbose:
                print(f"Pruned: infeasaible or failed ({result['message']})")
            continue

        x = result["x"]
        bound_value = result["objective"]

        if verbose:
            print(f"LP solution: {np.round(x, 6)}")
            print(f"Bound value: {bound_value:.6f}")

        if maximize:
            if bound_value <= best_obj + EPS:
                if verbose:
                    print(f"Pruned by bound: {bound_value:.6f} <= {best_obj:.6f}")
                continue
        else:
            if bound_value >= best_obj - EPS:
                if verbose:
                    print(f"Pruned by bound: {bound_value:.6f} >= {best_obj:.6f}")
                continue

        if is_integer_solution(x, integer_indices):
            if maximize:
                if bound_value > best_obj + EPS:
                    best_obj = bound_value
                    best_x = x.copy()
                    if verbose:
                        print(f"New best integer solution: {best_obj:.6f}")

            else:
                if bound_value < best_obj - EPS:
                    best_obj = bound_value
                    best_x = x.copy()
                    if verbose:
                        print(f"New best integer solution: {best_obj:.6f}")
            continue

        j = choose_branch_variable(x, integer_indices)
        if j is None:
            continue

        v = x[j]
        left_ub = floor(v)
        right_lb = ceil(v)

        if verbose:
            print(f"Branch on x[{j}] = {v:.6f}")
            print(f"Left child : x[{j}] <= {left_ub}")
            print(f"Right child: x[{j}] >= {right_lb}")

        left_bounds = node_bounds.copy()
        lb, ub = left_bounds[j]
        new_ub = left_ub if ub is None else min(ub, left_ub)
        left_bounds[j] = (lb, new_ub)

        right_bounds = node_bounds.copy()
        lb, ub = right_bounds[j]
        new_lb = right_lb if lb is None else max(lb, right_lb)
        right_bounds[j] = (new_lb, ub)

        if left_bounds[j][1] is None or left_bounds[j][0] is None or left_bounds[j][0] <= left_bounds[j][1]:
            stack.append({
                "bounds": left_bounds,
                "depth": node["depth"] + 1,
                "name": f"{node['name']}_L"
            })
        if right_bounds[j][1] is None or right_bounds[j][0] is None or right_bounds[j][0] <= right_bounds[j][1]:
            stack.append({
                "bounds": right_bounds,
                "depth": node["depth"] + 1,
                "name": f"{node['name']}_R"
            })

    return {
        "best_objective": None if best_x is None else best_obj,
        "best_x": best_x,
        "visited_nodes": visited_nodes
    }


def make_problem(name, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                 bounds=None, integer_indices=None, binary_indices=None,
                 maximize=True):
    return {
        "name": name,
        "c": np.array(c, dtype=float),
        "A_ub": None if A_ub is None else np.array(A_ub, dtype=float),
        "b_ub": None if b_ub is None else np.array(b_ub, dtype=float),
        "A_eq": None if A_eq is None else np.array(A_eq, dtype=float),
        "b_eq": None if b_eq is None else np.array(b_eq, dtype=float),
        "bounds": bounds,
        "integer_indices": integer_indices if integer_indices is not None else [],
        "binary_indices": binary_indices if binary_indices is not None else [],
        "maximize": maximize
    }


def generate_binary_problems():
    return [
        make_problem("B1", [8, 6, 5, 9], [[2, 1, 1, 3], [1, 2, 0, 1]], [5, 3],
                     bounds=[(0, 1)] * 4, integer_indices=[0, 1, 2, 3], binary_indices=[0, 1, 2, 3]),
        make_problem("B2", [4, 7, 3, 6, 5], [[1, 2, 1, 3, 2], [2, 0, 1, 1, 1]], [6, 4],
                     bounds=[(0, 1)] * 5, integer_indices=[0, 1, 2, 3, 4], binary_indices=[0, 1, 2, 3, 4]),
        make_problem("B3", [10, 12, 8], [[3, 4, 2], [1, 1, 1]], [7, 2],
                     bounds=[(0, 1)] * 3, integer_indices=[0, 1, 2], binary_indices=[0, 1, 2]),
        make_problem("B4", [5, 11, 4, 7], [[2, 3, 1, 2], [1, 0, 1, 1]], [5, 2],
                     bounds=[(0, 1)] * 4, integer_indices=[0, 1, 2, 3], binary_indices=[0, 1, 2, 3]),
        make_problem("B5", [9, 5, 6, 4, 7], [[2, 1, 2, 1, 3], [0, 1, 1, 1, 1]], [6, 3],
                     bounds=[(0, 1)] * 5, integer_indices=[0, 1, 2, 3, 4], binary_indices=[0, 1, 2, 3, 4]),
    ]


def generate_pure_integer_problems():
    return [
        make_problem("P1", [3, 2], [[2, 1], [1, 2]], [8, 8],
                     bounds=[(0, None), (0, None)], integer_indices=[0, 1]),
        make_problem("P2", [5, 4, 6], [[1, 2, 1], [3, 1, 2]], [9, 12],
                     bounds=[(0, None)] * 3, integer_indices=[0, 1, 2]),
        make_problem("P3", [7, 3, 5], [[2, 1, 2], [1, 3, 1]], [10, 9],
                     bounds=[(0, None)] * 3, integer_indices=[0, 1, 2]),
        make_problem("P4", [6, 8], [[3, 2], [1, 4]], [12, 11],
                     bounds=[(0, None)] * 2, integer_indices=[0, 1]),
        make_problem("P5", [4, 9, 5], [[2, 3, 1], [1, 1, 2]], [11, 8],
                     bounds=[(0, None)] * 3, integer_indices=[0, 1, 2]),
    ]


def generate_mixed_integer_problems():
    return [
        make_problem("M1", [5, 4, 3], [[2, 3, 1], [1, 1, 1]], [10, 7],
                     bounds=[(0, None), (0, None), (0, 1)], integer_indices=[1, 2], binary_indices=[2]),
        make_problem("M2", [6, 5, 8], [[1, 2, 3], [2, 1, 1]], [9, 8],
                     bounds=[(0, None)] * 3, integer_indices=[1]),
        make_problem("M3", [7, 2, 6, 4], [[2, 1, 2, 1], [1, 3, 1, 2]], [11, 10],
                     bounds=[(0, None), (0, None), (0, None), (0, 1)], integer_indices=[0, 3], binary_indices=[3]),
        make_problem("M4", [3, 9, 5], [[1, 2, 2], [2, 1, 1]], [8, 7],
                     bounds=[(0, None)] * 3, integer_indices=[1]),
        make_problem("M5", [4, 6, 7, 2], [[1, 2, 1, 3], [2, 1, 2, 1]], [10, 9],
                     bounds=[(0, None), (0, None), (0, None), (0, 1)], integer_indices=[2, 3], binary_indices=[3]),
    ]


def run_problem(problem):
    result = branch_and_bound(
        c=problem["c"],
        A_ub=problem["A_ub"],
        b_ub=problem["b_ub"],
        A_eq=problem["A_eq"],
        b_eq=problem["b_eq"],
        bounds=problem["bounds"],
        integer_indices=problem["integer_indices"],
        binary_indices=problem["binary_indices"],
        maximize=problem["maximize"],
        verbose=False
    )

    print(f"\n{problem['name']}")
    print(f"objective = {result['best_objective']}")
    print(f"x = {None if result['best_x'] is None else np.round(result['best_x'], 4)}")
    print(f"visited_nodes = {result['visited_nodes']}")


def main():
    binary_problems = generate_binary_problems()
    pure_integer_problems = generate_pure_integer_problems()
    mixed_integer_problems = generate_mixed_integer_problems()

    print("=" * 60)
    print("BINARY INTEGER PROGRAMMING")
    print("=" * 60)
    for p in binary_problems:
        run_problem(p)

    print("\n" + "=" * 60)
    print("PURE INTEGER PROGRAMMING")
    print("=" * 60)
    for p in pure_integer_problems:
        run_problem(p)

    print("\n" + "=" * 60)
    print("MIXED INTEGER PROGRAMMING")
    print("=" * 60)
    for p in mixed_integer_problems:
        run_problem(p)


if __name__ == "__main__":
    main()
