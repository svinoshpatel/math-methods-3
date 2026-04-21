import numpy as np
import numpy.typing as npt
from typing import Any
import json

ArrayF = npt.NDArray[np.float64]

def load_tasks(filename="tasks.json"):
    with open(filename, "r") as f:
        return json.load(f)

def transportation_simplex(
    costs: ArrayF,
    supply: ArrayF,
    demand: ArrayF
):
    costs, supply, demand = balance_problem(costs, supply, demand)
    rows, cols = costs.shape

    plan, basis = northwest_corner(supply, demand)

    print("Started optimization")
    iteration = 0
    while True:
        iteration += 1

        current_cost = np.sum(plan * costs)
        print(f"Iteration {iteration}: Current cost Z = {current_cost}")

        u, v = compute_potentials(costs, basis, rows, cols)

        deltas = compute_deltas(u, v, costs)
        
        non_basic_mask = np.ones(deltas.shape, dtype=bool)
        for r, c in basis:
            non_basic_mask[r, c] = False

        if np.all(deltas[non_basic_mask] <= 1e-9):
            total_cost = np.sum(plan * costs)
            print(f"Optimal plan was found at {iteration} iteration")
            print(f"Minimal cost: {total_cost}")
            return plan, total_cost
        
        temp_deltas = deltas.copy()
        temp_deltas[~non_basic_mask] = -np.inf
        entering = choose_entering_cell(temp_deltas)

        cycle = find_cycle(basis, entering)

        basis = update_plan(plan, cycle, basis)

def balance_problem(
    costs: ArrayF,
    supply: ArrayF,
    demand: ArrayF
) -> tuple[ArrayF, ArrayF, ArrayF]:
    if np.any(costs < 0) or np.any(supply < 0) or np.any(demand < 0):
        raise ValueError("Input data contains negative values.")
    
    diff = np.sum(demand) - np.sum(supply)
    
    if diff > 0:
        dummy_row = np.zeros((1, costs.shape[1]))
        costs = np.vstack([costs, dummy_row])
        supply = np.append(supply, diff)
    elif diff < 0:
        dummy_col = np.zeros((costs.shape[0], 1))
        costs = np.hstack([costs, dummy_col])
        demand = np.append(demand, abs(diff))
        
    return costs, supply, demand

def northwest_corner(
    supply: ArrayF,
    demand: ArrayF
) -> tuple[ArrayF, list[tuple[int, int]]]:
    s = supply.copy()
    d = demand.copy()

    rows, cols = len(s), len(d)
    plan = np.zeros((rows, cols))
    basis: list[tuple[int, int]] = []

    i = j = 0
    while i < rows and j < cols:
        basis.append((i, j))
        alloc = min(s[i], d[j])
        plan[i, j] = alloc

        s[i] -= alloc
        d[j] -= alloc

        if s[i] == 0 and i < rows - 1:
            i += 1
        else:
            j += 1

    return plan, basis

def compute_potentials(
    costs: ArrayF,
    basis: list[tuple[int, int]],
    rows: int,
    cols: int
) -> tuple[ArrayF, ArrayF]:
    u = np.full(rows, np.nan)
    v = np.full(cols, np.nan)

    u[0] = 0.0

    while np.isnan(u).any() or np.isnan(v).any():
        for (i, j) in basis:
            if not np.isnan(u[i]) and np.isnan(v[j]):
                v[j] = costs[i, j] - u[i]
            elif not np.isnan(v[j]) and np.isnan(u[i]):
                u[i] = costs[i, j] - v[j]

    return u, v 

def compute_deltas(
    u: ArrayF,
    v: ArrayF,
    costs: ArrayF
) -> ArrayF:
    potential_sum_matrix = u[:, np.newaxis] + v
    deltas = potential_sum_matrix - costs

    return deltas

def choose_entering_cell(
    deltas: ArrayF
) -> tuple[int, int]:
    i, j = np.unravel_index(np.argmax(deltas), deltas.shape)

    return i, j

def find_cycle(
    basis: list[tuple[int, int]],
    entering: tuple[int, int],
) -> list[tuple[int, int]]:
    cells = basis + [entering]
    while True:
        initial_count = len(cells)
        rows_counts = {}
        cols_counts = {}
        for r, c in cells:
            rows_counts[r] = rows_counts.get(r, 0) + 1
            cols_counts[c] = cols_counts.get(c, 0) + 1

        cells = [(r, c) for r, c in cells if rows_counts[r] > 1 and cols_counts[c] > 1]
        
        if len(cells) == initial_count:
            break

    ordered_cycle = [entering]
    while len(ordered_cycle) < len(cells):
        curr_r, curr_c = ordered_cycle[-1]
        for next_r, next_c in cells:
            if (next_r, next_c) not in ordered_cycle:
                is_row_step = (len(ordered_cycle) % 2 == 1)
                if is_row_step and curr_r == next_r:
                    ordered_cycle.append((next_r, next_c))
                    break
                elif not is_row_step and curr_c == next_c:
                    ordered_cycle.append((next_r, next_c))
                    break
    
    return ordered_cycle

def update_plan(
    plan: ArrayF,
    cycle: list[tuple[int, int]],
    basis: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    plus_cells = cycle[0::2]
    minus_cells = cycle[1::2]

    theta = min(plan[r, c] for r, c in minus_cells)

    for r, c in plus_cells:
        plan[r, c] += theta
    for r, c in minus_cells:
        plan[r, c] -= theta

    new_basis = list(basis)
    new_basis.append(cycle[0])

    for r, c in minus_cells:
        if plan[r, c] < 1e-9:
            new_basis.remove((r, c))
            plan[r, c] = 0.0
            break

    return new_basis

if __name__ == "__main__":
    tasks = load_tasks("tasks.json")
    for task in tasks:
        print(f"Task {task['id']} (Size: {len(task['supply'])}x{len(task['demand'])})")

        c = np.array(task['costs'], dtype=float)
        s = np.array(task['supply'], dtype=float)
        d = np.array(task['demand'], dtype=float)

        try:

            final_plan, final_cost = transportation_simplex(c, s, d)

            if c.shape[0] < 10:
                print("Оптимальний план:")
                print(final_plan)

        except Exception as e:
            print(f"Exception on task {task['id']}: {e}")

        print()
        print()

