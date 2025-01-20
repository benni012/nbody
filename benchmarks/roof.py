import matplotlib.pyplot as plt
import numpy as np


# Constants
# https://www.techpowerup.com/gpu-specs/geforce-mx150.c2959
t_perf = 1_117 * 10**9  # Theoretical FP32performance in FLOPs/s
t_mem_bw = 48.06 * 10**9  # Theoretical memory bandwidth in bytes/s

# Kernel Parameters
N = 5e3  # Number of particles
kernel_time = 1266 / 1e6 # Kernel execution time in seconds
# memory_traffic = 4 * 4 * N  # Memory traffic in bytes
kernel_flops = 49312746  # 1140135792  # Kernel FLOPs

# Compute Operational Intensity (OI) and Achieved Performance
# oi = kernel_flops / memory_traffic  # Operational Intensity (FLOPs per byte)
achieved_perf = kernel_flops / kernel_time  # Achieved Performance (FLOPs/s)
dr = 5765  # 0
dw = 2478
oi = kernel_flops/(dw + dr) * 32

# Define Roofline Model
oi_vals = np.logspace(-5, 5, 100)  # Operational intensity values (log scale)
memory_roof = t_mem_bw * oi_vals  # Memory bandwidth roofline
compute_roof = [t_perf] * len(oi_vals)  # Compute roofline

# Plot Roofline Model
plt.figure(figsize=(10, 6))
plt.loglog(oi_vals, memory_roof, label='Memory Bandwidth Roof', linestyle='--')
plt.loglog(oi_vals, compute_roof, label='Compute Capability Roof', linestyle='--')

# Plot Achieved Performance
plt.scatter([oi], [achieved_perf], color='red', label='Achieved Performance')

# Add Labels and Legend
plt.xlabel("Operational Intensity (FLOPs/byte)")
plt.ylabel("Performance (FLOPs/s)")
plt.title("Empirical Roofline Model")
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.ylim(10**6, t_perf * 1.1)  # Adjust limits for better visualization
# plt.xlim(10**-2, 10**2)
plt.show()
