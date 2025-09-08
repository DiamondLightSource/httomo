import cupy as cp

arr = cp.array([1, 2, 3])
print(f"arr before: {arr}")
arr += 1
print(f"arr after: {arr}")
