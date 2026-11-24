"""
MATRIX ROADMAP — PURE PYTHON + NUMPY
====================================
A single runnable study file covering the practical programming side of the
matrix roadmap: matrix basics/types, operations, determinants, minors,
cofactors, inverse, row reduction, rank, linear systems, eigenvalues,
decompositions, numerical methods, and ML matrix examples.

Requirements:
    pip install numpy

Run:
    python matrix.py

Notes:
- "Pure Python" implementations use lists and loops (no third-party package).
- NumPy implementations use ndarray and numpy.linalg.
- Floating-point answers may have tiny rounding differences.
"""

from math import sqrt
import numpy as np


# ============================================================
# 0. SHARED HELPERS
# ============================================================


def shape(A):
    """Return (rows, columns); reject ragged matrices."""
    if not A:
        return (0, 0)
    cols = len(A[0])
    if any(len(row) != cols for row in A):
        raise ValueError("Matrix rows must all have the same length.")
    return len(A), cols


def print_matrix(A, title=None):
    if title:
        print(f"\n{title}")
    for row in A:
        print(" ".join(f"{float(x):10.4g}" for x in row))


def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def zeros(rows, cols):
    return [[0 for _ in range(cols)] for _ in range(rows)]


def to_numpy(A):
    return np.array(A, dtype=float)


# ============================================================
# PART 1 — MATRIX BASICS AND TYPES
# ============================================================


def matrix_basics_demo():
    A = [[1, 2, 3], [4, 5, 6]]
    m, n = shape(A)
    print("\nPART 1 — BASICS")
    print("A =", A)
    print("Order:", (m, n))
    print("Element a[0][1]:", A[0][1])  # Python uses zero-based indexing.
    print("Rows:", m, "Columns:", n)

    examples = {
        "row matrix": [[1, 2, 3]],
        "column matrix": [[2], [5], [8]],
        "rectangular matrix": [[1, 2, 3], [4, 5, 6]],
        "square matrix": [[1, 2], [3, 4]],
        "zero matrix": zeros(2, 3),
        "diagonal matrix": [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
        "scalar matrix": [[5, 0], [0, 5]],
        "identity matrix": identity(3),
        "upper triangular": [[1, 2, 3], [0, 4, 5], [0, 0, 6]],
        "lower triangular": [[1, 0, 0], [2, 3, 0], [4, 5, 6]],
        "singular matrix": [[1, 2], [2, 4]],
        "non-singular matrix": [[1, 2], [3, 4]],
        "symmetric matrix": [[1, 2], [2, 3]],
        "skew-symmetric matrix": [[0, 2], [-2, 0]],
    }
    for name, M in examples.items():
        print_matrix(M, name)

    # Orthogonal matrix: Q.T @ Q = I.
    theta = np.pi / 4
    Q = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    print_matrix(Q, "orthogonal rotation matrix")
    print("Q.T @ Q =\n", np.array(Q).T @ np.array(Q))


# ============================================================
# PART 2 — PURE PYTHON MATRIX OPERATIONS
# ============================================================


def matrix_equal(A, B, tol=1e-9):
    if shape(A) != shape(B):
        return False
    return all(
        abs(A[i][j] - B[i][j]) <= tol for i in range(len(A)) for j in range(len(A[0]))
    )


def add(A, B):
    if shape(A) != shape(B):
        raise ValueError("Addition requires equal dimensions.")
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def subtract(A, B):
    if shape(A) != shape(B):
        raise ValueError("Subtraction requires equal dimensions.")
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def scalar_multiply(k, A):
    return [[k * value for value in row] for row in A]


def transpose(A):
    rows, cols = shape(A)
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def matmul(A, B):
    rows_a, cols_a = shape(A)
    rows_b, cols_b = shape(B)
    if cols_a != rows_b:
        raise ValueError("A columns must equal B rows.")
    C = zeros(rows_a, cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_a))
    return C


def dot(u, v):
    if len(u) != len(v):
        raise ValueError("Vectors must have equal lengths.")
    return sum(x * y for x, y in zip(u, v))


def matrix_operations_demo():
    print("\nPART 2 — PURE PYTHON OPERATIONS")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    print("Equal:", matrix_equal(A, [[1, 2], [3, 4]]))
    print_matrix(add(A, B), "A + B")
    print_matrix(subtract(A, B), "A - B")
    print_matrix(scalar_multiply(2, A), "2A")
    print_matrix(matmul(A, B), "A @ B")
    print_matrix(transpose(A), "A transpose")
    print("Dot product [1,2] · [3,4] =", dot([1, 2], [3, 4]))


# ============================================================
# PART 3 — DETERMINANTS, MINORS, COFACTORS, ADJUGATE
# ============================================================


def minor_matrix(A, row_to_remove, col_to_remove):
    return [
        [A[i][j] for j in range(len(A)) if j != col_to_remove]
        for i in range(len(A))
        if i != row_to_remove
    ]


def determinant(A):
    n, m = shape(A)
    if n != m:
        raise ValueError("Determinant requires a square matrix.")
    if n == 0:
        return 1
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    # Laplace expansion along the first row.
    return sum(
        ((-1) ** j) * A[0][j] * determinant(minor_matrix(A, 0, j)) for j in range(n)
    )


def minor(A, i, j):
    return determinant(minor_matrix(A, i, j))


def cofactor(A, i, j):
    return ((-1) ** (i + j)) * minor(A, i, j)


def cofactor_matrix(A):
    n, m = shape(A)
    if n != m:
        raise ValueError("Cofactor matrix requires square matrix.")
    return [[cofactor(A, i, j) for j in range(n)] for i in range(n)]


def adjugate(A):
    return transpose(cofactor_matrix(A))


def determinant_demo():
    print("\nPART 3 — DETERMINANTS")
    A = [[1, 2, 3], [0, 4, 5], [1, 0, 6]]
    print("det(A):", determinant(A))
    print("Minor M[0,0]:", minor(A, 0, 0))
    print_matrix(cofactor_matrix(A), "Cofactor matrix")
    print_matrix(adjugate(A), "Adjugate / adjoint")
    B = [[1, 2], [3, 4]]
    print("det([[1,2],[3,4]]):", determinant(B))


# ============================================================
# PART 4 — INVERSE AND ELEMENTARY ROW OPERATIONS
# ============================================================


def inverse_adjugate(A, tol=1e-12):
    det = determinant(A)
    if abs(det) < tol:
        raise ValueError("Singular matrix: inverse does not exist.")
    return scalar_multiply(1 / det, adjugate(A))


def swap_rows(A, i, j):
    A[i], A[j] = A[j], A[i]


def scale_row(A, i, factor):
    A[i] = [factor * x for x in A[i]]


def add_scaled_row(A, target, source, factor):
    A[target] = [x + factor * y for x, y in zip(A[target], A[source])]


def rref(A, tol=1e-12):
    """Return reduced row echelon form and pivot-column indices."""
    M = [list(map(float, row)) for row in A]
    rows, cols = shape(M)
    pivot_cols = []
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break
        best = max(range(pivot_row, rows), key=lambda r: abs(M[r][col]))
        if abs(M[best][col]) < tol:
            continue
        swap_rows(M, pivot_row, best)

        pivot = M[pivot_row][col]
        scale_row(M, pivot_row, 1.0 / pivot)

        for r in range(rows):
            if r != pivot_row:
                factor = -M[r][col]
                add_scaled_row(M, r, pivot_row, factor)

        pivot_cols.append(col)
        pivot_row += 1

    M = [[0.0 if abs(x) < tol else x for x in row] for row in M]
    return M, pivot_cols


def inverse_gauss_jordan(A, tol=1e-12):
    n, m = shape(A)
    if n != m:
        raise ValueError("Inverse requires a square matrix.")
    augmented = [list(map(float, A[i])) + identity(n)[i] for i in range(n)]
    reduced, pivots = rref(augmented, tol)
    if pivots[:n] != list(range(n)):
        raise ValueError("Singular matrix: inverse does not exist.")
    return [row[n:] for row in reduced]


def rank(A, tol=1e-12):
    return len(rref(A, tol)[1])


def inverse_and_row_reduction_demo():
    print("\nPART 4 — INVERSE / ROW REDUCTION")
    A = [[1, 2], [3, 4]]
    print_matrix(inverse_adjugate(A), "Inverse by adjugate")
    print_matrix(inverse_gauss_jordan(A), "Inverse by Gauss-Jordan")
    print_matrix(rref([[1, 2, 3], [2, 4, 6]])[0], "RREF")
    print("Rank:", rank([[1, 2, 3], [2, 4, 6]]))


# ============================================================
# PART 5 — LINEAR SYSTEMS, GAUSSIAN ELIMINATION, CRAMER
# ============================================================


def solve_gaussian(A, b, tol=1e-12):
    """Solve Ax=b by pivoted Gauss-Jordan elimination (unique solution)."""
    n, m = shape(A)
    if n != m or len(b) != n:
        raise ValueError("This solver expects square A and matching b.")
    aug = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    reduced, pivots = rref(aug, tol)
    for row in reduced:
        if all(abs(x) < tol for x in row[:-1]) and abs(row[-1]) >= tol:
            raise ValueError("Inconsistent system: no solution.")
    if len([p for p in pivots if p < n]) < n:
        raise ValueError("System does not have a unique solution.")
    return [reduced[i][-1] for i in range(n)]


def solve_cramers(A, b, tol=1e-12):
    n, m = shape(A)
    if n != m:
        raise ValueError("Cramer's rule requires square A.")
    det_a = determinant(A)
    if abs(det_a) < tol:
        raise ValueError("det(A)=0; Cramer's rule cannot give a unique solution.")
    answer = []
    for col in range(n):
        replaced = [row[:] for row in A]
        for i in range(n):
            replaced[i][col] = b[i]
        answer.append(determinant(replaced) / det_a)
    return answer


def linear_system_demo():
    print("\nPART 5 — LINEAR SYSTEMS")
    # x + 2y + 0z = -1
    # 3x + 4y + z = 2
    # -2x + 3y + 2z = 5
    A = [[1, 2, 0], [3, 4, 1], [-2, 3, 2]]
    b = [-1, 2, 5]
    print("Gaussian solution [x,y,z]:", solve_gaussian(A, b))
    print("NumPy solution:", np.linalg.solve(to_numpy(A), np.array(b, dtype=float)))

    A2 = [[1, 1], [2, 3]]
    b2 = [5, 8]
    print("Cramer's rule [x,y]:", solve_cramers(A2, b2))
    print("Homogeneous system AX=0 always has trivial solution X=0.")
    print("For a non-homogeneous system, consistency can be checked with ranks.")


# ============================================================
# PART 6 — NUMPY MATRIX OPERATIONS AND LINEAR ALGEBRA
# ============================================================


def numpy_demo():
    print("\nPART 6 — NUMPY")
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    print("A:\n", A)
    print("shape:", A.shape)
    print("A+B:\n", A + B)
    print("A-B:\n", A - B)
    print("2*A:\n", 2 * A)
    print("Elementwise multiply A*B:\n", A * B)
    print("Matrix multiply A@B:\n", A @ B)
    print("Transpose:\n", A.T)
    print("det(A):", np.linalg.det(A))
    print("inverse(A):\n", np.linalg.inv(A))
    print("rank(A):", np.linalg.matrix_rank(A))
    print("solve Ax=b:", np.linalg.solve(A, np.array([5.0, 11.0])))
    print("eigenvalues:", np.linalg.eigvals(A))
    print("eigenvalues + eigenvectors:", np.linalg.eig(A))
    print("Singular Value Decomposition U, s, Vt:")
    U, s, Vt = np.linalg.svd(A)
    print(U, s, Vt, sep="\n")
    print("QR decomposition Q, R:")
    Q, R = np.linalg.qr(A)
    print(Q, R, sep="\n")
    print("Cholesky (for symmetric positive-definite matrices):")
    P = np.array([[4.0, 2.0], [2.0, 3.0]])
    print(np.linalg.cholesky(P))


# ============================================================
# PART 7 — EIGENVALUES, EIGENVECTORS, DIAGONALIZATION
# ============================================================


def eigen_demo():
    print("\nPART 7 — EIGENVALUES")
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    values, vectors = np.linalg.eig(A)
    print("A:\n", A)
    print("Eigenvalues:", values)
    print("Eigenvectors (columns):\n", vectors)
    for i, lam in enumerate(values):
        v = vectors[:, i]
        print(f"Check eigenpair {i}: A@v =", A @ v, "lambda*v =", lam * v)

    # For a diagonalizable matrix: A = P D P^-1.
    P = vectors
    D = np.diag(values)
    print("Reconstructed P@D@inv(P):\n", P @ D @ np.linalg.inv(P))


# ============================================================
# PART 8 — ADVANCED LINEAR ALGEBRA
# ============================================================


def advanced_demo():
    print("\nPART 8 — ADVANCED LINEAR ALGEBRA")
    # Vector space / basis / dimension: standard basis of R^3.
    basis = np.eye(3)
    print("Standard basis of R^3:\n", basis)
    print("Dimension:", basis.shape[1])

    u = np.array([1.0, 2.0, 0.0])
    v = np.array([2.0, -1.0, 0.0])
    print("Dot product:", np.dot(u, v))
    print("Orthogonal:", np.isclose(np.dot(u, v), 0))

    # Projection of vector b onto the column space of A.
    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 2.0, 2.0])
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    projection = A @ coeffs
    print("Least-squares coefficients:", coeffs)
    print("Orthogonal projection:", projection)

    # Gram-Schmidt (columns of input matrix).
    def gram_schmidt(X):
        Q_cols = []
        for j in range(X.shape[1]):
            w = X[:, j].astype(float).copy()
            for q in Q_cols:
                w -= np.dot(q, w) * q
            norm = np.linalg.norm(w)
            if norm > 1e-12:
                Q_cols.append(w / norm)
        return np.column_stack(Q_cols) if Q_cols else np.empty((X.shape[0], 0))

    X = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    Q = gram_schmidt(X)
    print("Gram-Schmidt orthonormal columns:\n", Q)
    print("Q.T@Q:\n", Q.T @ Q)

    # Spectral decomposition for real symmetric A.
    S = np.array([[2.0, 1.0], [1.0, 2.0]])
    eigenvalues, V = np.linalg.eigh(S)
    spectral_reconstruction = V @ np.diag(eigenvalues) @ V.T
    print("Spectral reconstruction:\n", spectral_reconstruction)


# ============================================================
# PART 9 — NUMERICAL METHODS
# ============================================================


def power_method(A, iterations=100, tol=1e-10):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    x = np.ones(n)
    x /= np.linalg.norm(x)
    old_value = 0.0
    for _ in range(iterations):
        y = A @ x
        norm = np.linalg.norm(y)
        if norm == 0:
            return 0.0, x
        x = y / norm
        value = float(x @ (A @ x))
        if abs(value - old_value) < tol:
            break
        old_value = value
    return value, x


def jacobi_solver(A, b, iterations=100, tol=1e-10):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if np.any(np.isclose(np.diag(A), 0)):
        raise ValueError("Jacobi method requires non-zero diagonal entries.")
    x = np.zeros_like(b)
    for _ in range(iterations):
        x_new = (b - (A @ x) + np.diag(A) * x) / np.diag(A)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x


def gauss_seidel(A, b, iterations=100, tol=1e-10):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
    for _ in range(iterations):
        old = x.copy()
        for i in range(n):
            left = np.dot(A[i, :i], x[:i])
            right = np.dot(A[i, i + 1 :], old[i + 1 :])
            if abs(A[i, i]) < 1e-12:
                raise ValueError("Zero diagonal entry.")
            x[i] = (b[i] - left - right) / A[i, i]
        if np.linalg.norm(x - old, ord=np.inf) < tol:
            break
    return x


def numerical_methods_demo():
    print("\nPART 9 — NUMERICAL METHODS")
    A = [[4.0, 1.0], [2.0, 3.0]]
    b = [1.0, 2.0]
    print("Power method:", power_method([[2.0, 1.0], [1.0, 2.0]]))
    print("Jacobi:", jacobi_solver(A, b))
    print("Gauss-Seidel:", gauss_seidel(A, b))
    print("Reference NumPy solution:", np.linalg.solve(np.array(A), np.array(b)))


# ============================================================
# PART 10 — MATRIX CONCEPTS IN MACHINE LEARNING
# ============================================================


def machine_learning_demo():
    print("\nPART 10 — MATRIX IN MACHINE LEARNING")
    # Feature matrix: rows=samples, columns=features.
    X = np.array(
        [
            [1000.0, 2.0],
            [1500.0, 3.0],
            [2000.0, 4.0],
            [2500.0, 5.0],
        ]
    )
    y = np.array([200.0, 300.0, 400.0, 500.0])
    print("Feature matrix X:\n", X)

    # Add intercept column and solve least-squares normal equation.
    X_design = np.column_stack([X, np.ones(X.shape[0])])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    predictions = X_design @ beta
    print("Coefficient vector beta:", beta)
    print("Predictions:", predictions)

    # Covariance matrix: rows are observations, columns are variables.
    print("Covariance matrix:\n", np.cov(X, rowvar=False))

    # Weight matrix / linear layer.
    W = np.array([[0.1, 0.2], [0.3, 0.4]])
    features = np.array([[1.0, 2.0]])
    print("Neural-network linear layer output:", features @ W)

    # Transformation matrix (2D rotation).
    angle = np.pi / 6
    T = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    point = np.array([1.0, 0.0])
    print("Transformed point:", T @ point)

    # PCA using SVD (center the data first).
    centered = X - X.mean(axis=0)
    U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
    principal_components = Vt.T
    projected = centered @ principal_components
    print("PCA principal directions:\n", principal_components)
    print("PCA projected data:\n", projected)

    # Embedding / recommendation matrices are ordinary 2D arrays.
    embeddings = np.array([[0.2, 0.1, 0.7], [0.5, 0.4, 0.1]])
    user_item_ratings = np.array([[5.0, 0.0, 3.0], [0.0, 4.0, 0.0]])
    print("Embedding matrix:\n", embeddings)
    print("User-item recommendation matrix:\n", user_item_ratings)


# ============================================================
# PART 11 — IMPORTANT THEOREMS (NUMERICAL CHECKS)
# ============================================================


def theorem_checks_demo():
    print("\nPART 11 — THEOREM CHECKS")
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    # Cayley-Hamilton: a matrix satisfies its own characteristic polynomial.
    # For 2x2: A^2 - trace(A)A + det(A)I = 0.
    lhs = A @ A - np.trace(A) * A + np.linalg.det(A) * np.eye(2)
    print("Cayley-Hamilton residual (near zero):\n", lhs)

    # Rank-nullity: rank(A) + nullity(A) = number of columns.
    B = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    r = np.linalg.matrix_rank(B)
    nullity = B.shape[1] - r
    print("Rank:", r, "Nullity:", nullity, "Columns:", B.shape[1])

    # Invertible matrix theorem: nonzero determinant iff invertible.
    C = np.array([[1.0, 2.0], [3.0, 4.0]])
    print("det(C):", np.linalg.det(C))
    print("C @ inv(C):\n", C @ np.linalg.inv(C))

    # Spectral theorem: real symmetric matrices have orthonormal eigenvectors.
    eigenvalues, Q = np.linalg.eigh(A)
    print("Symmetric matrix eigenvalues:", eigenvalues)
    print("Q.T @ Q:\n", Q.T @ Q)
    print("Q @ diag(lambda) @ Q.T:\n", Q @ np.diag(eigenvalues) @ Q.T)


# ============================================================
# PART 12 — ARRAY VS MATRIX AND COMPLEXITY NOTES
# ============================================================


def programming_notes():
    print("\nPART 12 — PROGRAMMING NOTES")
    print("Python nested list: flexible list-of-lists representation.")
    print("NumPy ndarray: efficient n-dimensional homogeneous array.")
    print("For numerical work, prefer ndarray over numpy.matrix.")
    print("Typical complexities:")
    print("  Addition/subtraction: O(m*n)")
    print("  Scalar multiplication: O(m*n)")
    print("  Transpose: O(m*n) to create a copy")
    print("  Matrix multiplication (basic): O(m*n*p)")
    print("  Determinant by Laplace expansion: O(n!) (mainly educational)")
    print("  Gaussian elimination: O(n^3)")
    print("  Dense inverse: O(n^3)")
    print("  NumPy uses optimized low-level numerical libraries where available.")


# ============================================================
# RUN ALL DEMOS
# ============================================================


def main():
    matrix_basics_demo()
    matrix_operations_demo()
    determinant_demo()
    inverse_and_row_reduction_demo()
    linear_system_demo()
    numpy_demo()
    eigen_demo()
    advanced_demo()
    numerical_methods_demo()
    machine_learning_demo()
    theorem_checks_demo()
    programming_notes()


if __name__ == "__main__":
    main()
