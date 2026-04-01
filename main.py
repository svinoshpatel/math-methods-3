import numpy as np

EPSILON = 1e-9
BIG_M = 1e9

def solve(c_orig, A_orig, b_orig, constraint_types, verbose=True):
    # m - number of constrants (rows), n - number of original vars (cols)
    m, n = A_orig.shape

    # 1. Standardize: Convert all b to positive
    for i in range(m):
        if b_orig[i] < 0:
            A_orig[i] *= -1
            b_orig[i] *= -1
            if constraint_types[i] == '<=': constraint_types[i] = '>='
            elif constraint_types[i] == '>=': constraint_types[i] = '<='

    # 2. Add Slack and Artificial Variables
    # We need to track which variables are original, slack, or artificial
    A_extended = A_orig.copy()
    c_extended = list(c_orig)
    
    for i, c_type in enumerate(constraint_types):
        if c_type == '<=':
            # Add one slack variable
            col = np.zeros((m, 1))
            col[i] = 1
            A_extended = np.hstack([A_extended, col])
            c_extended.append(0.0)
        elif c_type == '>=':
            # Add one surplus variable and one artificial variable
            col_s = np.zeros((m, 1))
            col_s[i] = -1
            col_a = np.zeros((m, 1))
            col_a[i] = 1
            A_extended = np.hstack([A_extended, col_s, col_a])
            c_extended.append(0.0) # Surplus
            c_extended.append(-BIG_M) # Artificial (Penalty for MAX)
        elif c_type == '=':
            # Add one artificial variable
            col_a = np.zeros((m, 1))
            col_a[i] = 1
            A_extended = np.hstack([A_extended, col_a])
            c_extended.append(-BIG_M) # Artificial

    # 3. Initialize Basis
    # The basis must consist of m variables that form an Identity matrix.
    # This will be our slack (for <=) and artificial (for >= and =) variables.
    num_total_vars = A_extended.shape[1]
    basis = []
    for i in range(m):
        # Find the column that is 1 at index i and 0 elsewhere (Identity column)
        found = False
        for j in range(n, num_total_vars):
            if abs(A_extended[i, j] - 1.0) < EPSILON and np.sum(np.abs(A_extended[:, j])) < 1.0 + EPSILON:
                basis.append(j)
                found = True
                break
        if not found: raise ValueError(f"Could not find initial basis for row {i}")

    A = A_extended
    c = np.array(c_extended)
    b = np.array(b_orig)
    Binv = np.eye(m) # create identity matrix for slacks
    xB = b.copy()
    iterations = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"SETUP")
        print(f"  Variables (n):    {n}")
        print(f"  Constraints (m): {m}")
        print(f"  Initial basis indices: {basis}")
        print(f"  Initial xB:      {xB}")
        print(f"{'='*60}")

    while True:
        # cB is coefficients of basis vars (includes -BIG_M for artificials)
        cB = c[basis]
        pi = cB @ Binv  # shadow price / simplex multiplier

        # full col from original matrix
        def full_col(j):
            return A[:, j]

        reduced_costs = [
            # c[j] is what we gain directly, pi @ full_col(j) - indirect loss
            c[j] - pi @ full_col(j)
            for j in range(num_total_vars)
        ]

        # if reduced_cost positive = profit outweighs loss (for Maximize)
        enter = int(np.argmax(reduced_costs))

        if verbose: # Limit verbose output
            print(f"\nIteration {iterations}")
            print(f"  Basis variables: {basis}")
            print(f"  xB (values):      {np.round(xB, 4)}")
            print(f"  Current Obj:      {sum(c_orig[j]*xB[i] for i,j in enumerate(basis) if j < n):.2f}")

        if reduced_costs[enter] <= EPSILON:
            break # optimal solution found

        aBar = Binv @ full_col(enter) # pivot col

        ratios = np.where(aBar > EPSILON, xB / aBar, np.inf)
        leave = int(np.argmin(ratios))

        if ratios[leave] == np.inf:
            # problem in problem forumulation
            raise ValueError("Problem is unbounded.")

        pivot = aBar[leave]
        if abs(pivot) < EPSILON:
            # duplicate constraints
            raise ValueError("Degenerate pivot.")

        # Update Binv and xB using row operations
        Binv[leave] /= pivot
        for i in range(m):
            if i != leave:
                Binv[i] -= aBar[i] * Binv[leave]

        xB[leave] /= pivot
        for i in range(m):
            if i != leave:
                xB[i] -= aBar[i] * (xB[leave])
        
        basis[leave] = enter
        iterations += 1

        if iterations > 1000:
            raise ValueError("Iteration limit reached.")

    # Calculate final objective using only original variables
    obj = sum(c_orig[j] * xB[i] for i, j in enumerate(basis) if j < n)
    return obj, xB, basis, iterations

if __name__ == "__main__":
    c = [150, 180, 210, 130, 250, 190, 160, 220, 140, 200, 175, 230]
    A = [
        [12,12,12,12,12,12, 15,15,15,15,15,15],  # 1
        [2,2,2,2, 3,3,3,3, 2.5,2.5,2.5,2.5],     # 2
        [5,8,5,8,5,8,5,8,5,8,5,8],               # 3
        [1,1,1,0,0,0,0,0,0,0,0,0],               # 4
        [-1,-1,0,0,1,0,0,0,0,0,0,0],             # 5
        [0,0,0,-2,0,0,0,0,0,0,0,1],              # 6
        [0,0,0,0,0,0,0,0,0,1,0,0],               # 7
        [0,0,0,0,0,1,-1,0,0,0,0,0],              # 8
        [0,0,0,0,0,0,0,1,1,0,1,0],               # 9
        [1,0,1,0,1,0,1,0,1,0,1,0],               # 10
        [1,1,1,1,1,1,1,1,1,1,1,1],               # 11
        [0,1,0,0,0,0,0,0,-1,0,0,0],              # 12
        [0,0,0,1,0,0,0,1,0,0,0,0],               # 13
        [-1,0,-1,0,0,0,0,0,0,0,1,0]              # 14
    ]
    b = [2000, 400, 1500, 15, 0, 0, 25, 0, 60, 40, 180, 180, 10, 0]
    # Note: Ensure the 'types' match the number of rows in A
    types = ['<=', '<=', '<=', '>=', '<=', '>=', '<=', '>=', '<=', '>=', '<=', '<=', '>=', '<=']
    
    try:
        obj, xB, basis, iters = solve(c, np.array(A), b, types)
        print(f"\nOptimization Successful!")
        print(f"Objective Value: {obj:.2f}")
        print(f"Iterations: {iters}")

        results = [0.0] * 12
        for val, idx in zip(xB, basis):
            if idx < 12:
                results[idx] = val

        print("\nVariable Values:")
        for i, val in enumerate(results):
            print(f"x{i+1}: {val:.4f}")

    except ValueError as e:
        print(f"Error: {e}")
