import ctypes 
import numpy as np 
import time 
lib = ctypes.cdll.LoadLibrary("./libmatrix.so") 

lib.gpu_matrix_multiply.argtypes = [ 
 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"), 
 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"), 
 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"), 
 ctypes.c_int 
] 
N = 1024 
A = np.random.rand(N, N).astype(np.float32) 
B = np.random.rand(N, N).astype(np.float32) 
C = np.zeros((N, N), dtype=np.float32) 
start = time.time() 
lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N) 
end = time.time() 

print(f"time: {end-before}") 

