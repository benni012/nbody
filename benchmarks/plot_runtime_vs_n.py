import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import numpy as np

directory = "."

data = []

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Extract N, BH, and GPU information from the filename using regex
        match = re.search(r"N(\d+)_BH(\d+)_GPU(\d+)", filename)
        if match:
            N = int(match.group(1))
            BH = int(match.group(2))
            GPU = int(match.group(3))
        
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        
        for _, row in df.iterrows():
            data.append({
                "N": N,
                "BH": BH,
                "GPU": GPU,
                "Function": row["Function"],
                "Mean Time (us)": row["Mean Time (us)"]
            })

results_df = pd.DataFrame(data)

colors = {0: "blue", 1: "orange"}

plt.figure(figsize=(12, 8))

# Unique functions and markers
markers = ['o', 's', 'D', 'v', 'x', '^', '<', '>', 'p', '*']  # Markers for different functions

for gpu in [0, 1]:
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
        alpha=0.9
    )
    
    for i, function in enumerate(unique_functions[::-1]):
        func_df = gpu_df[gpu_df["Function"] == function]
        func_df = func_df.sort_values(by="N")
        func_df["Mean Time (us)"] = func_df["Mean Time (us)"].clip(1., None)
        plt.plot(
            func_df["N"],
            func_df["Mean Time (us)"],
            marker=markers[i % len(markers)],
            linestyle="--",
            color=colors[gpu],
            label=f"GPU {gpu} - {function}",
            alpha=0.35
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
