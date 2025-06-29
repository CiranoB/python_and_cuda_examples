import cupy as cp
import numpy as np

# It's necessary to import the linalg submodule from cupy.scipy
# This requires SciPy to be installed in the environment.
from scipy import linalg as cupy_linalg

if __name__ == '__main__':
    # --- Create a sample square matrix on the GPU using CuPy ---
    A_gpu = np.array([
        [2, 5, 8, 7],
        [5, 2, 2, 8],
        [7, 5, 6, 6],
        [5, 4, 4, 8]
    ], dtype=np.float64)

    print("Original Matrix A:\n", A_gpu)
    print("-" * 40)

    p, l, u = cupy_linalg.lu(A_gpu)
    np.allclose(A_gpu, p @ l @ u)

    p, _, _ = cupy_linalg.lu(A_gpu, p_indices=True)

    print(np.allclose(A_gpu, l[p, :] @ u))

    print("\n--- LDU Decomposition Results ---")
    print("\nP (Permutation Matrix):\n", p)
    print("\nL (Lower Triangular with 1s on diagonal):\n", l)
    print("\nU' (Upper Triangular with 1s on diagonal):\n", u)


