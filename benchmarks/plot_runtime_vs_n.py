import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import bench_utils as utils

directory = "."

results_df = utils.get_benchmark_df(".")
# results_df= results_df[results_df["BH"] == True] 
results_df= results_df[results_df["GPU"] == True] 

colors = {0: "blue", 1: "orange"}

plt.figure(figsize=(12, 8))

# Unique functions and markers

for gpu in [True, False]:
    gpu_df = results_df[results_df["GPU"] == gpu]
    unique_functions = gpu_df["Function"].unique()  # Functions specific to this GPU

    total_runtime = gpu_df.groupby("N")["Mean Time (us)"].sum().reset_index()
    plt.plot(
        total_runtime["N"],
        total_runtime["Mean Time (us)"],
        color=colors[gpu],
        linestyle="-",
        marker="o",
        label=f"GPU {gpu} - Total Runtime",
        alpha=0.9,
    )

    for i, function in enumerate(unique_functions[::-1]):
        func_df = gpu_df[gpu_df["Function"] == function]
        func_df = func_df.sort_values(by="N")
        func_df["Mean Time (us)"] = func_df["Mean Time (us)"].clip(1.0, None)
        plt.plot(
            func_df["N"],
            func_df["Mean Time (us)"],
            marker=utils.markers[i % len(utils.markers)],
            linestyle="--",
            color=colors[gpu],
            label=f"GPU {gpu} - {function}",
            alpha=0.35,
        )

plt.xlabel("N (Number of Bodies)")
plt.ylabel("Runtime (us)")
plt.title("Runtime Breakdown by Function: GPU 0 vs GPU 1")
plt.legend(loc="best", fontsize="small")
plt.grid(True)
# plt.yscale("log")
# plt.xscale("log")
plt.tight_layout()
plt.show()
