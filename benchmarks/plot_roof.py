import matplotlib.pyplot as plt
import numpy as np
import bench_utils as utils
import matplotlib.cm as cm
import matplotlib.colors as mcolors

results_df = utils.get_benchmark_df(".")
results_df = results_df[results_df["GPU"] == True].drop(columns="GPU")
nv_df = utils.get_nv_profs(".")
print(nv_df)
print(results_df)

# Constants
# https://www.techpowerup.com/gpu-specs/geforce-mx150.c2959
t_perf = 1_117 * 10**9  # Theoretical FP32 performance in FLOPs/s
t_mem_bw = 48.06 * 10**9  # Theoretical memory bandwidth in bytes/s

# Kernel Parameters
fig, ax = plt.subplots()
# Plot Roofline Model
oi_vals = np.logspace(-5, 8, 100)  # Operational intensity values (log scale)
# Define Roofline Model
memory_roof = t_mem_bw * oi_vals  # Memory bandwidth roofline
compute_roof = [t_perf] * len(oi_vals)  # Compute roofline
ax.loglog(oi_vals[:-50], memory_roof[:-50], label='Memory Bandwidth Roof', linestyle='--')
ax.loglog(oi_vals[-51:], compute_roof[-51:], label='Compute Capability Roof', linestyle='--')

# Create a colormap
N_vals = np.sort(nv_df['N'].unique())
norm = mcolors.LogNorm(vmin=N_vals.min(), vmax=N_vals.max())
colormap_bh = cm.autumn  # viridis inferno plasma
colormap_naive = cm.winter

kernel_names = nv_df["Kernel"].unique()
for a in [True, False]:
    a_results_df = results_df[results_df["BH"] == a]
    a_nv_df = nv_df[nv_df["BH"] == a]
    N_vals = np.sort(a_nv_df['N'].unique())
    for i, N in enumerate(N_vals):
        results_rows = a_results_df[a_results_df["N"] == N ]
        nv_rows = a_nv_df[a_nv_df["N"] == N]
        kernel_time = results_rows[results_rows["Function"].isin(kernel_names)]["Mean Time (us)"].sum() / 1e3
        kernel_flops = nv_rows["flop_count_sp_Avg_Value"].sum()  # Kernel FLOPs

        # Compute Operational Intensity (OI) and Achieved Performance
        achieved_perf = kernel_flops / kernel_time  # Achieved Performance (FLOPs/s)
        dr = nv_rows["dram_read_throughput_Avg_Value"].sum()
        dw = nv_rows["dram_write_throughput_Avg_Value"].sum() if a else 0 # TODO wich is 0
        oi = kernel_flops / (dw + dr) * 32

        color = colormap_bh(norm(N)) if a else colormap_naive(norm(N))
        # Plot Achieved Performance
        ax.scatter([oi], [achieved_perf], color=color, label=f'N={N}', marker=utils.markers[i % len(utils.markers)])

# Add a colorbar for the colormap
# sm = cm.ScalarMappable(cmap=colormap, norm=norm)
# sm.set_array([])  # Needed for colorbar
# cbar = fig.colorbar(sm, ax=ax)
# cbar.set_label("N (log scale)")

# Add Labels and Legend
ax.set_xlabel("Operational Intensity (FLOPs/byte)")
ax.set_ylabel("Performance (FLOPs/s)")
ax.set_title("Empirical Roofline Model")
# ax.legend(ncol=10, loc="best")
ax.grid(True, which="both", linestyle='--', linewidth=0.5)
# ax.set_ylim(10**6, t_perf * 1.1)  # Adjust limits for better visualization
# ax.set_xlim(10**-2, 10**2)
plt.tight_layout()
plt.show()
