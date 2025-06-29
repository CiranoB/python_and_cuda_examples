import numpy as np
import time

# --- Optional Dependency Imports ---
# We will try to import GPU libraries and set flags for availability.

try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# --- Multiplication Functions ---

def numpy_mult(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Performs matrix multiplication using NumPy on the CPU.
    """
    print("--- Starting NumPy Multiplication ---")
    start_time = time.time()
    
    result_matrix = np.dot(matrix_a, matrix_b)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"NumPy multiplication took: {elapsed_time:.6f} seconds")

    return result_matrix, elapsed_time

def cupy_mult(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    Performs matrix multiplication using CuPy on the GPU.
    CuPy abstracts away the kernel and memory management.
    """
    if not CUPY_AVAILABLE:
        print("\nCuPy not found. Skipping CuPy multiplication.")
        return None, float('inf')

    print("\n--- Starting CuPy Multiplication ---")
    start_time = time.time()

    matrix_a_gpu = cp.asarray(matrix_a)
    matrix_b_gpu = cp.asarray(matrix_b)

    result_gpu = cp.dot(matrix_a_gpu, matrix_b_gpu)

    result_matrix = cp.asnumpy(result_gpu)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"CuPy multiplication (including memory transfer) took: {elapsed_time:.6f} seconds")
    
    return result_matrix, elapsed_time

def pycuda_mult(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple[np.ndarray | None, float]:
    """
    Performs matrix multiplication using PyCUDA on the GPU by writing a custom kernel.
    """
    if not PYCUDA_AVAILABLE:
        print("\nPyCUDA not found. Skipping PyCUDA multiplication.")
        return None, float('inf')

    print("\n--- Starting PyCUDA Multiplication ---")
    
    # Measure total time including data transfers and kernel execution
    total_start_time = time.time()

    # --- 1. Transfer data from Host (CPU) to Device (GPU) ---
    matrix_a_gpu = gpuarray.to_gpu(matrix_a)
    matrix_b_gpu = gpuarray.to_gpu(matrix_b)
    result_gpu = gpuarray.empty((matrix_a.shape[0], matrix_b.shape[1]), np.float32)

    # --- 2. Define the CUDA C Kernel ---
    mod = SourceModule("""
    __global__ void mat_mult_kernel(float *A, float *B, float *C, int width_a, int width_b)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0f;
        if (row < width_a && col < width_b) {
            for (int i = 0; i < width_a; ++i) {
                sum += A[row * width_a + i] * B[i * width_b + col];
            }
            C[row * width_b + col] = sum;
        }
    }
    """)
    cuda_kernel = mod.get_function("mat_mult_kernel")

    # --- 3. Launch the Kernel on the GPU ---
    BLOCK_SIZE = (32, 32, 1)
    grid_x = (result_gpu.shape[1] + BLOCK_SIZE[0] - 1) // BLOCK_SIZE[0]
    grid_y = (result_gpu.shape[0] + BLOCK_SIZE[1] - 1) // BLOCK_SIZE[1]
    GRID_SIZE = (grid_x, grid_y)
    
    cuda_kernel(
        matrix_a_gpu, matrix_b_gpu, result_gpu,
        np.int32(matrix_a.shape[1]), np.int32(matrix_b.shape[1]),
        block=BLOCK_SIZE, grid=GRID_SIZE
    )

    # Wait for the kernel to finish and transfer result back
    drv.Context.synchronize()
    result_matrix = result_gpu.get()
    
    total_end_time = time.time()
    elapsed_time = total_end_time - total_start_time
    print(f"PyCUDA multiplication (including memory transfer) took: {elapsed_time:.6f} seconds")
    
    return result_matrix, elapsed_time


if __name__ == '__main__':
    # For larger sizes (e.g., 1024x1024+), the GPU's parallelism shows its strength.
    MATRIX_SIZE = 15000

    print(f"Creating two {MATRIX_SIZE}x{MATRIX_SIZE} matrices (float32)...")
    
    # Initialize matrices with 32-bit floats for GPU computation
    matrix_1 = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    matrix_2 = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    
    # --- Run and time each version ---
    numpy_result, numpy_time = numpy_mult(matrix_1, matrix_2)
    pycuda_result, pycuda_time = pycuda_mult(matrix_1, matrix_2)
    cupy_result, cupy_time = cupy_mult(matrix_1, matrix_2)
    
    # --- Final Summary and Verification ---
    print("\n--- Comparison Summary ---")
    print(f"NumPy elapsed time:    {numpy_time:.6f} seconds")
    if PYCUDA_AVAILABLE:
        print(f"PyCUDA elapsed time:   {pycuda_time:.6f} seconds")
    if CUPY_AVAILABLE:
        print(f"CuPy elapsed time:     {cupy_time:.6f} seconds")

    print("\n--- Verification ---")
    verified_pycuda = False
    if pycuda_result is not None:
        if np.allclose(numpy_result, pycuda_result, atol=1e-2):
            print("Verification successful: NumPy and PyCUDA results are close.")
            verified_pycuda = True
        else:
            print("Verification FAILED: NumPy and PyCUDA results do not match!")

    verified_cupy = False
    if cupy_result is not None:
        if np.allclose(numpy_result, cupy_result, atol=1e-2):
            print("Verification successful: NumPy and CuPy results are close.")
            verified_cupy = True
        else:
            print("Verification FAILED: NumPy and CuPy results do not match!")
