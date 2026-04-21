import numpy as np

EPSILON = 1e-9
BIG_M = 1e9


def solve(c_orig, A_orig, b_orig, constraint_types, verbose=False):
    A_orig = np.array(A_orig, dtype=float)
    b_orig = np.array(b_orig, dtype=float)
    c_orig = np.array(c_orig, dtype=float)
    constraint_types = list(constraint_types)

    m, n = A_orig.shape

    # 1. Make all RHS values non-negative
    for i in range(m):
        if b_orig[i] < 0:
            A_orig[i] *= -1
            b_orig[i] *= -1
            if constraint_types[i] == "<=":
                constraint_types[i] = ">="
            elif constraint_types[i] == ">=":
                constraint_types[i] = "<="

    # 2. Add slack / surplus / artificial variables
    A_ext = A_orig.copy()
    c_ext = list(c_orig)

    for i, ctype in enumerate(constraint_types):
        if ctype == "<=":
            col = np.zeros((m, 1))
            col[i] = 1
            A_ext = np.hstack([A_ext, col])
            c_ext.append(0.0)
        elif ctype == ">=":
            col_s = np.zeros((m, 1)); col_s[i] = -1
            col_a = np.zeros((m, 1)); col_a[i] = 1
            A_ext = np.hstack([A_ext, col_s, col_a])
            c_ext.append(0.0)       # surplus
            c_ext.append(-BIG_M)    # artificial (Big-M penalty)
        elif ctype == "=":
            col_a = np.zeros((m, 1)); col_a[i] = 1
            A_ext = np.hstack([A_ext, col_a])
            c_ext.append(-BIG_M)    # artificial

    # 3. Build initial basis from identity columns (slacks / artificials)
    num_vars = A_ext.shape[1]
    basis = []
    for i in range(m):
        for j in range(n, num_vars):
            if abs(A_ext[i, j] - 1.0) < EPSILON and np.sum(np.abs(A_ext[:, j])) < 1.0 + EPSILON:
                basis.append(j)
                break
        else:
            raise ValueError(f"Could not find initial basis column for row {i}")

    A = A_ext
    c = np.array(c_ext)
    b = b_orig.copy()
    Binv = np.eye(m)
    xB   = b.copy()
    iters = 0

    if verbose:
        print(f"\n{'='*60}\nSETUP: n={n}, m={m}")
        print(f"  Initial basis indices: {basis}")
        print(f"  Initial xB: {xB}")
        print("="*60)

    # 4. Simplex iterations (Fundamental Insight: B^{-1} updated via row operations)
    while True:
        cB = c[basis]
        pi = cB @ Binv          # simplex multipliers (shadow prices)

        # Reduced costs: r_j = c_j - pi @ a_j
        rc = np.array([c[j] - pi @ A[:, j] for j in range(num_vars)])
        enter = int(np.argmax(rc))

        if verbose:
            obj_cur = sum(float(c_orig[j]) * float(xB[i]) for i, j in enumerate(basis) if j < n)
            print(f"Iter {iters}: basis={basis}, xB={np.round(xB,4)}, z={obj_cur:.4f}")

        if rc[enter] <= EPSILON:
            break   # all reduced costs <= 0 => optimal

        aBar = Binv @ A[:, enter]       # representation of entering column in current basis
        ratios = np.where(aBar > EPSILON, xB / aBar, np.inf)
        leave = int(np.argmin(ratios))

        if ratios[leave] == np.inf:
            raise ValueError("Problem is unbounded.")

        pivot = aBar[leave]
        if abs(pivot) < EPSILON:
            raise ValueError("Degenerate pivot encountered.")

        # Update B^{-1} and xB via elementary row operations
        Binv[leave] /= pivot
        xB[leave]   /= pivot
        for i in range(m):
            if i != leave:
                Binv[i] -= aBar[i] * Binv[leave]
                xB[i]   -= aBar[i] * xB[leave]

        basis[leave] = enter
        iters += 1
        if iters > 10_000:
            raise ValueError("Iteration limit (10000) exceeded — check problem formulation.")

    obj = sum(float(c_orig[j]) * float(xB[i]) for i, j in enumerate(basis) if j < n)
    return obj, xB, basis, iters

BASE_c = [5.0, 4.0, 3.0]
BASE_A = [
    [6, 4, 2],
    [3, 5, 5],
    [1, 2, 4],
]
BASE_b = [240.0, 270.0, 100.0]
BASE_types = ["<=", "<=", "<="]


def extract_solution(xB, basis, n_orig):
    x = [0.0] * n_orig
    for val, idx in zip(xB, basis):
        if idx < n_orig:
            x[idx] = float(val)
    return x


def sensitivity_analysis():
    n = len(BASE_c)

    obj0, xB0, basis0, iters0 = solve(
        BASE_c, [r[:] for r in BASE_A], BASE_b[:], BASE_types[:]
    )
    x0 = extract_solution(xB0, basis0, n)

    print("=" * 65)
    print("БАЗОВИЙ РОЗВ'ЯЗОК")
    print(f"  x* = {[round(v, 4) for v in x0]}")
    print(f"  z* = {obj0:.4f}")
    print(f"  Кількість ітерацій: {iters0}")
    print("=" * 65)

    print("\n" + "─" * 65)
    print("1. АНАЛІЗ ЧУТЛИВОСТІ ДО c (коефіцієнти цільової функції)")
    print("─" * 65)

    for vi in range(n):
        base_val = BASE_c[vi]
        print(f"\n  Варіація c[{vi+1}] (базове значення = {base_val}):")
        print(f"  {'c_'+str(vi+1):>8}  {'z*':>12}  {'x*':>30}")
        print(f"  {'─'*55}")
        for delta in np.linspace(-base_val, base_val * 3, 15):
            new_c = BASE_c[:]
            new_c[vi] = base_val + delta
            try:
                obj, xB, basis, _ = solve(
                    new_c, [r[:] for r in BASE_A], BASE_b[:], BASE_types[:], verbose=False
                )
                x = extract_solution(xB, basis, n)
                marker = " <-- base" if abs(delta) < 0.01 else ""
                print(f"  {new_c[vi]:>8.2f}  {obj:>12.4f}  {[round(v,2) for v in x]}{marker}")
            except ValueError as e:
                print(f"  {new_c[vi]:>8.2f}  Error: {e}")

    print("\n" + "─" * 65)
    print("2. АНАЛІЗ ЧУТЛИВОСТІ ДО b (праві частини обмежень)")
    print("─" * 65)

    for ri in range(len(BASE_b)):
        base_val = BASE_b[ri]
        print(f"\n  Варіація b[{ri+1}] (базове значення = {base_val}):")
        print(f"  {'b_'+str(ri+1):>8}  {'z*':>12}  {'x*':>30}")
        print(f"  {'─'*55}")
        for delta in np.linspace(-base_val * 0.8, base_val * 1.5, 15):
            new_b = BASE_b[:]
            new_b[ri] = base_val + delta
            try:
                obj, xB, basis, _ = solve(
                    BASE_c[:], [r[:] for r in BASE_A], new_b, BASE_types[:], verbose=False
                )
                x = extract_solution(xB, basis, n)
                marker = " <-- base" if abs(delta) < 0.5 else ""
                print(f"  {new_b[ri]:>8.2f}  {obj:>12.4f}  {[round(v,2) for v in x]}{marker}")
            except ValueError as e:
                print(f"  {new_b[ri]:>8.2f}  Error: {e}")

    print("\n" + "─" * 65)
    print("3. АНАЛІЗ ЧУТЛИВОСТІ ДО A (матриця коефіцієнтів обмежень)")
    print("─" * 65)
    print("   (Варіюємо A[i][1] — коефіцієнт при x1 в кожному обмеженні)")

    for ri in range(len(BASE_A)):
        ci = 0  # coefficient of x1
        base_val = BASE_A[ri][ci]
        print(f"\n  Варіація A[{ri+1}][{ci+1}] (базове значення = {base_val}):")
        print(f"  {'A_'+str(ri+1)+str(ci+1):>8}  {'z*':>12}  {'x*':>30}")
        print(f"  {'─'*55}")
        for delta in np.linspace(-base_val * 0.8, base_val * 1.5, 12):
            new_A = [r[:] for r in BASE_A]
            new_A[ri][ci] = base_val + delta
            try:
                obj, xB, basis, _ = solve(
                    BASE_c[:], new_A, BASE_b[:], BASE_types[:], verbose=False
                )
                x = extract_solution(xB, basis, n)
                marker = " <-- base" if abs(delta) < 0.05 else ""
                print(f"  {new_A[ri][ci]:>8.2f}  {obj:>12.4f}  {[round(v,2) for v in x]}{marker}")
            except ValueError as e:
                print(f"  {new_A[ri][ci]:>8.2f}  Error: {e}")


if __name__ == "__main__":
    sensitivity_analysis()
