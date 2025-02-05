import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import bench_utils as utils

directory = "."

results_df = utils.get_benchmark_df(".")
# results_df= results_df[results_df["BH"] == False] 
# results_df= results_df[results_df["GPU"] == True] 

# #00c7fd
# #0068b5
colors = {1: "#76b900", 0: "#0068b5"}



for bh in [True, False]:
    bh_char = "Barnes-Hut" if bh else "Naive"
    algo_df = results_df[results_df["BH"] == bh]
    fig,ax = plt.subplots()
    ax.plot([algo_df["N"].min(), algo_df["N"].max()], [1e6/60] * 2, linestyle=":", color="tab:red", label="60 FPS threshold")
    for gpu in [True, False]:
        gpu_char = "GPU" if gpu else "CPU"
        gpu_df = algo_df[algo_df["GPU"] == gpu]
        unique_functions = gpu_df["Function"].unique()  # Functions specific to this GPU

        total_runtime = gpu_df.groupby("N")["Mean Time (us)"].sum().reset_index()
        ax.plot(
            total_runtime["N"],
            total_runtime["Mean Time (us)"],
            color=colors[gpu],
            linestyle="-",
            marker="o",
            label=f"{gpu_char} - Total Runtime",
            alpha=0.9,
        )

        for i, function in enumerate(unique_functions[::-1]):
            func_df = gpu_df[gpu_df["Function"] == function]
            func_df = func_df.sort_values(by="N")
            func_df["Mean Time (us)"] = func_df["Mean Time (us)"].clip(1.0, None)
            ax.plot(
                func_df["N"],
                func_df["Mean Time (us)"],
                marker=utils.markers[i % len(utils.markers)],
                linestyle="--",
                color=colors[gpu],
                label=f"{gpu_char} - {function}",
                alpha=0.35,
            )
    ax.set_xlabel("N (Number of Bodies)")
    ax.set_ylabel("Runtime (us)")
    ax.set_title(f"{bh_char}: GPU vs CPU")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True)
    # plt.yscale("log")
    # plt.xscale("log")

for gpu in [True, False]:
    gpu_char = "GPU" if gpu else "CPU"
    gpu_df = results_df[results_df["GPU"] == gpu]
    fig,ax = plt.subplots(1, 2, sharey=True)
    for bh in [True, False]:
        bh_char = "Barnes-Hut" if bh else "Naive"
        algo_df = gpu_df[gpu_df["BH"] == bh]
        ax[int(bh)].plot([algo_df["N"].min(), algo_df["N"].max()], [1e6/60] * 2, linestyle=":", color="tab:red", label="60 FPS threshold")
        unique_functions = algo_df["Function"].unique()  # Functions specific to this GPU

        total_runtime = algo_df.groupby("N")["Mean Time (us)"].sum().reset_index()
        ax[int(bh)].plot(
            total_runtime["N"],
            total_runtime["Mean Time (us)"],
            color=colors[int(gpu)],
            linestyle="-",
            marker="o",
            label=f"{bh_char} - Total Runtime",
            alpha=0.9,
        )

        for i, function in enumerate(unique_functions[::-1]):
            func_df = algo_df[algo_df["Function"] == function]
            func_df = func_df.sort_values(by="N")
            func_df["Mean Time (us)"] = func_df["Mean Time (us)"].clip(1.0, None)
            ax[int(bh)].plot(
                func_df["N"],
                func_df["Mean Time (us)"],
                marker=utils.markers[i % len(utils.markers)],
                linestyle="--",
                color=colors[int(gpu)],
                label=f"{bh_char} - {function}",
                alpha=0.35,
            )

        ax[int(bh)].set_xlabel("N (Number of Bodies)")
        ax[int(bh)].set_ylabel("Runtime (us)")
        ax[int(bh)].set_title(f"{bh_char} on {gpu_char}")
        ax[int(bh)].legend(loc="best", fontsize="small")
        ax[int(bh)].grid(True)
    # plt.yscale("log")
    # plt.xscale("log")

plt.tight_layout()
plt.show()
