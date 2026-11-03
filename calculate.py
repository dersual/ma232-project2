"""
MA 232 Course Project 2: Pattern Recognition Calculations
Calculates matrices M and N, verifies all 65,536 binary states,
and generates the LaTeX verification table for Part C.
"""

import itertools
import numpy as np

# Part A: Target pattern vector w (16-dimensional, row-major)
# Row 1: [1, 1, 0, 1]
# Row 2: [0, 0, 1, 0]
# Row 3: [1, 0, 0, 1]
# Row 4: [1, 0, 1, 0]
w = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], dtype=int)

blue_indices = np.where(w == 1)[0]
white_indices = np.where(w == 0)[0]


def build_matrix_m():
    """
    Constructs 16x16 matrix M using the basis of W_perp (Module 4, pages 32-36).
    Solving w^T x = 0 with x_0 as the pivot variable:
      x_0 = -(x_1 + x_3 + x_6 + x_8 + x_11 + x_12 + x_14)
    """
    basis_vectors = []
    # 8 basis vectors for coordinates where w_k == 0 (x_0 = 0)
    for idx in white_indices:
        v = np.zeros(16, dtype=int)
        v[idx] = 1
        basis_vectors.append(v)

    # 7 basis vectors for other coordinates where w_k == 1 (x_0 = -1)
    pivot = blue_indices[0]
    for idx in blue_indices[1:]:
        v = np.zeros(16, dtype=int)
        v[pivot] = -1
        v[idx] = 1
        basis_vectors.append(v)

    B = np.array(basis_vectors, dtype=int)  # 15 x 16 matrix
    M = B.T @ B  # 16 x 16 symmetric matrix
    return B, M


def build_matrix_n():
    """
    Constructs secondary 16x16 matrix N using orthogonal projection onto W_perp:
      N = 8 * I - w * w^T
    """
    N = 8 * np.eye(16, dtype=int) - np.outer(w, w)
    return N


def verify_all_binary_vectors(matrix, name="M"):
    """
    Exhaustively tests all 2^16 = 65,536 binary vectors.
    Returns the vectors that yield Q(u) == 0.
    """
    all_patterns = np.array(list(itertools.product([0, 1], repeat=16)), dtype=int)
    # Vectorized Q(u) = sum((u @ matrix) * u, axis=1)
    scores = np.sum((all_patterns @ matrix) * all_patterns, axis=1)
    zero_mask = scores == 0
    zero_vectors = all_patterns[zero_mask]
    nonzero_min = np.min(scores[~zero_mask])

    print(f"--- Verification for Matrix {name} ---")
    print(f"Total binary vectors tested: {len(all_patterns)}")
    print(f"Patterns where Q(u) == 0: {len(zero_vectors)}")
    for z in zero_vectors:
        is_zero = np.all(z == 0)
        is_target = np.all(z == w)
        print(f"  u = {z.tolist()} (All-zero: {is_zero}, Target w: {is_target})")
    print(f"Minimum nonzero Q(u) value: {nonzero_min}\n")
    return zero_vectors


def generate_sample_table(M, N):
    """
    Generates a sample table of test patterns required in Part C:
    - Target pattern w
    - 1 blue cell
    - 2 blue cells
    - 3 blue cells (different positions)
    - All 16 cells blue
    - Additional contrasting patterns
    """
    patterns = []

    # 1. Target vector w
    patterns.append(("Target Pattern $w$", w.copy()))

    # 2. Single blue cell
    p_single1 = np.zeros(16, dtype=int)
    p_single1[0] = 1  # Cell 1 (part of w)
    patterns.append(("Single blue cell (Cell 1)", p_single1))

    p_single2 = np.zeros(16, dtype=int)
    p_single2[2] = 1  # Cell 3 (not in w)
    patterns.append(("Single blue cell (Cell 3)", p_single2))

    # 3. Two blue cells
    p_two1 = np.zeros(16, dtype=int)
    p_two1[0] = 1
    p_two1[1] = 1  # Cells 1, 2 (both in w)
    patterns.append(("Two blue cells (Cells 1, 2)", p_two1))

    p_two2 = np.zeros(16, dtype=int)
    p_two2[0] = 1
    p_two2[2] = 1  # Cells 1 (in w), 3 (not in w)
    patterns.append(("Two blue cells (Cells 1, 3)", p_two2))

    # 4. Three blue cells in different positions
    p_three1 = np.zeros(16, dtype=int)
    p_three1[0] = 1
    p_three1[1] = 1
    p_three1[3] = 1  # Row 1 blue cells (all in w)
    patterns.append(("Three blue cells (Row 1: 1, 2, 4)", p_three1))

    p_three2 = np.zeros(16, dtype=int)
    p_three2[2] = 1
    p_three2[4] = 1
    p_three2[5] = 1  # Cells 3, 5, 6 (none in w)
    patterns.append(("Three blue cells (3, 5, 6)", p_three2))

    # 5. Half target (first 4 blue cells of w)
    p_half = np.zeros(16, dtype=int)
    p_half[blue_indices[:4]] = 1
    patterns.append(("Four target cells (1, 2, 4, 7)", p_half))

    # 6. Complement of w (inverted)
    p_comp = 1 - w
    patterns.append(("Inverted target ($1 - w$)", p_comp))

    # 7. All 16 cells blue
    p_all = np.ones(16, dtype=int)
    patterns.append(("All 16 cells blue", p_all))

    print("% --- LaTeX Table Output for Part C ---")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{clccc}")
    print("\\toprule")
    print("\\textbf{\\#} & \\textbf{Description} & \\textbf{Blue Count} & $Q_M(u) = u^T M u$ & $Q_N(u) = u^T N u$ \\\\")
    print("\\midrule")
    for i, (desc, u) in enumerate(patterns, 1):
        qm = int(u @ M @ u)
        qn = int(u @ N @ u)
        blue_cnt = int(np.sum(u))
        print(f"{i} & {desc} & {blue_cnt} & {qm} & {qn} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Sample test patterns and their quadratic form values.}")
    print("\\label{tab:test_patterns}")
    print("\\end{table}")


if __name__ == "__main__":
    print(f"Target vector w: {w.tolist()}")
    print(f"Blue cell count: {len(blue_indices)}, White cell count: {len(white_indices)}\n")

    B, M = build_matrix_m()
    N = build_matrix_n()

    print("Matrix M shape:", M.shape)
    print("Matrix M is symmetric:", np.all(M == M.T))
    print("Determinant of M:", np.linalg.det(M.astype(float)))
    print("Rank of M:", np.linalg.matrix_rank(M.astype(float)))
    print(f"w^T M w = {w @ M @ w}\n")

    print("Matrix N shape:", N.shape)
    print("Matrix N is symmetric:", np.all(N == N.T))
    print("Determinant of N:", np.linalg.det(N.astype(float)))
    print("Rank of N:", np.linalg.matrix_rank(N.astype(float)))
    print(f"w^T N w = {w @ N @ w}\n")

    verify_all_binary_vectors(M, "M")
    verify_all_binary_vectors(N, "N")

    generate_sample_table(M, N)
