"""
MA 232 Course Project 2: Pattern Recognition Calculations

This script handles the matrix math for Project 2:
1. Defines the target vector w from the 4x4 grid.
2. Builds Matrix M using the basis of W_perp (Module 4, pages 32-36).
3. Builds Matrix N using orthogonal complement projection (Module 4, pages 23-26).
4. Exhaustively checks all 2^16 = 65,536 binary vectors.
5. Computes quadratic form values for the sample test patterns in Part C.
"""

import itertools
import numpy as np

# ---------------------------------------------------------------------------
# 1. Target Vector Definition (Row-Major Ordering)
# ---------------------------------------------------------------------------
# Grid layout (4x4):
# Row 1: [Blue, Blue, White, Blue]  -> [1, 1, 0, 1]
# Row 2: [White, White, Blue, White]-> [0, 0, 1, 0]
# Row 3: [Blue, White, White, Blue] -> [1, 0, 0, 1]
# Row 4: [Blue, White, Blue, White] -> [1, 0, 1, 0]

w = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], dtype=int)
blue_indices = np.where(w == 1)[0]
white_indices = np.where(w == 0)[0]


# ---------------------------------------------------------------------------
# 2. Matrix M Construction (Module 4 Basis of W_perp Method)
# ---------------------------------------------------------------------------
def build_matrix_m():
    """
    Constructs 16x16 matrix M such that Null(M) = Span{w}.
    
    Solve w^T x = 0 with x_0 as the pivot variable:
      x_0 = -(x_1 + x_3 + x_6 + x_8 + x_11 + x_12 + x_14)
    
    Setting each of the 15 free variables to 1 (others 0) gives a basis
    for W_perp. Stacking them as rows forms a 15x16 matrix B.
    Then M = B^T B.
    """
    basis_vectors = []

    # 8 basis vectors for white cell indices (where w_k = 0, so x_0 = 0)
    for idx in white_indices:
        v = np.zeros(16, dtype=int)
        v[idx] = 1
        basis_vectors.append(v)

    # 7 basis vectors for other blue cell indices (where w_k = 1, so x_0 = -1)
    pivot = blue_indices[0]
    for idx in blue_indices[1:]:
        v = np.zeros(16, dtype=int)
        v[pivot] = -1
        v[idx] = 1
        basis_vectors.append(v)

    B = np.array(basis_vectors, dtype=int)
    M = B.T @ B
    return B, M


# ---------------------------------------------------------------------------
# 3. Matrix N Construction (Module 4 Orthogonal Projection Method)
# ---------------------------------------------------------------------------
def build_matrix_n():
    """
    Constructs 16x16 matrix N using orthogonal projection onto W_perp:
      P_perp = I - (w w^T) / (w^T w)
    
    Since w^T w = 8, scaling by 8 gives an all-integer matrix:
      N = 8*I - w w^T
    """
    N = 8 * np.eye(16, dtype=int) - np.outer(w, w)
    return N


# ---------------------------------------------------------------------------
# 4. Exhaustive Verification Across All 65,536 Binary Vectors
# ---------------------------------------------------------------------------
def verify_matrix(matrix, name="M"):
    """
    Tests every binary pattern u in {0, 1}^16.
    Verifies that Q(u) = u^T Matrix u == 0 ONLY for u = 0 and u = w.
    """
    all_patterns = np.array(list(itertools.product([0, 1], repeat=16)), dtype=int)
    scores = np.sum((all_patterns @ matrix) * all_patterns, axis=1)

    zero_mask = scores == 0
    zero_patterns = all_patterns[zero_mask]
    min_nonzero = np.min(scores[~zero_mask])

    print(f"--- Verification Results for Matrix {name} ---")
    print(f"Total binary patterns tested: {len(all_patterns):,}")
    print(f"Patterns where Q(u) == 0: {len(zero_patterns)}")
    for p in zero_patterns:
        is_zero = np.all(p == 0)
        is_target = np.all(p == w)
        print(f"  Pattern: {p.tolist()} (All-zero: {is_zero}, Target w: {is_target})")
    print(f"Lowest positive value for other patterns: {min_nonzero}\n")
    return zero_patterns


# ---------------------------------------------------------------------------
# 5. Sample Test Patterns for Part C Table
# ---------------------------------------------------------------------------
def calculate_sample_patterns(M, N):
    """
    Computes Q_M(u) and Q_N(u) for required test cases from Part C:
    - Target pattern w
    - Single blue cell
    - Two blue cells
    - Three blue cells
    - All 16 cells blue
    - Contrasting / inverted patterns
    """
    samples = []

    # 1. Target vector
    samples.append(("Target pattern $w$", w.copy()))

    # 2. Single blue cell
    s1 = np.zeros(16, dtype=int)
    s1[0] = 1
    samples.append(("Single blue cell (Cell 1, in $w$)", s1))

    s2 = np.zeros(16, dtype=int)
    s2[2] = 1
    samples.append(("Single blue cell (Cell 3, not in $w$)", s2))

    # 3. Two blue cells
    s3 = np.zeros(16, dtype=int)
    s3[0] = 1
    s3[1] = 1
    samples.append(("Two blue cells (Cells 1, 2, both in $w$)", s3))

    s4 = np.zeros(16, dtype=int)
    s4[0] = 1
    s4[2] = 1
    samples.append(("Two blue cells (Cells 1 in $w$, 3 not in $w$)", s4))

    # 4. Three blue cells in different positions
    s5 = np.zeros(16, dtype=int)
    s5[0] = 1
    s5[1] = 1
    s5[3] = 1
    samples.append(("Three blue cells (Row 1: 1, 2, 4 in $w$)", s5))

    s6 = np.zeros(16, dtype=int)
    s6[2] = 1
    s6[4] = 1
    s6[5] = 1
    samples.append(("Three blue cells (Cells 3, 5, 6, none in $w$)", s6))

    # 5. Four cells from target
    s7 = np.zeros(16, dtype=int)
    s7[blue_indices[:4]] = 1
    samples.append(("Four target cells (Cells 1, 2, 4, 7)", s7))

    # 6. Inverted target (white cells become blue, blue become white)
    s8 = 1 - w
    samples.append(("Inverted target ($1 - w$)", s8))

    # 7. All 16 cells blue
    s9 = np.ones(16, dtype=int)
    samples.append(("All 16 cells blue", s9))

    print("% --- LaTeX Table Rows for Part C ---")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{clccc}")
    print("\\toprule")
    print("\\textbf{\\#} & \\textbf{Description} & \\textbf{Blue Cells} & $Q_M(u) = u^T M u$ & $Q_N(u) = u^T N u$ \\\\")
    print("\\midrule")
    for i, (desc, u) in enumerate(samples, 1):
        qm = int(u @ M @ u)
        qn = int(u @ N @ u)
        blue_cnt = int(np.sum(u))
        print(f"{i} & {desc} & {blue_cnt} & {qm} & {qn} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Test patterns and quadratic-form evaluation values.}")
    print("\\label{tab:sample_patterns}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("MA 232 Project 2: Pattern Recognition Calculations\n")
    print(f"Target vector w: {w.tolist()}")
    print(f"Blue cells: {len(blue_indices)}, White cells: {len(white_indices)}\n")

    B, M = build_matrix_m()
    N = build_matrix_n()

    print("--- Matrix M Properties ---")
    print(f"Shape: {M.shape}")
    print(f"Symmetric: {np.array_equal(M, M.T)}")
    print(f"Rank: {np.linalg.matrix_rank(M.astype(float))} (out of 16)")
    print(f"Determinant: {np.linalg.det(M.astype(float)):.1f}")
    print(f"w^T M w: {w @ M @ w}\n")

    print("--- Matrix N Properties ---")
    print(f"Shape: {N.shape}")
    print(f"Symmetric: {np.array_equal(N, N.T)}")
    print(f"Rank: {np.linalg.matrix_rank(N.astype(float))} (out of 16)")
    print(f"Determinant: {np.linalg.det(N.astype(float)):.1f}")
    print(f"w^T N w: {w @ N @ w}\n")

    verify_matrix(M, "M")
    verify_matrix(N, "N")

    calculate_sample_patterns(M, N)
