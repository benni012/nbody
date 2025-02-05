import itertools
import subprocess
import re
import os

dir = os.path.abspath(os.getcwd())

start_pattern = r"==\d+== Profiling result:"
csv_start_pattern = r'"Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg"'


def filter_and_save_nvprof(input_text: str, file_name: str):
    csv_lines = []
    start_found = False
    for line in input_text.splitlines():
        if re.match(start_pattern, line):
            start_found = True
        if start_found and re.match(csv_start_pattern, line):
            csv_lines.append(line)
        elif start_found and line.startswith('"'):
            csv_lines.append(line)

    # Write to a CSV file
    with open(file_name, "w") as f:
        for line in csv_lines:
            f.write(line + "\n")


n_values = [
    10, 20, 30, 40, 50, 60, 70, 80, 90,
    100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3,
    1e4, 2e4, 3e4, 4e4,
    5e4, 6e4, 7e4, 8e4, 9e4,
    1e5, 2e5, 3e5, 4e5, 5e5
]
d_values = ["gpu", "cpu"]
a_values = ["naive"]

cpp_executable = "./nbody"

combinations = itertools.product(n_values, d_values, a_values)

for n, d, a in combinations:

    # RUNNING INTERNAL BENCHMARK FOR TIMING
    args = [cpp_executable, f"-n{n}", f"-d{d}", f"-a{a}", "-i1000", "-b"]
    print(f"Running: {args} in {dir}")
    try:
        result = subprocess.run(
            args,
            cwd=dir,
            # capture_output=False,
            # text=True,
            # check=True
        )
        print(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred for args: {args}")
        print(f"Error message: {e.stderr}")

    if d == "gpu":
        # RUNNING NVPROF BENCHMARK FOR MEM THROUGHPUT AND FP COUNT
        nvargs = [
            "sudo", 
            "nvprof",
            "--metrics",
            "flop_count_sp",
            "--metrics",
            "dram_read_throughput,dram_write_throughput",
            "--csv",
            "--print-gpu-summary",
            cpp_executable,
            f"-n{n}",
            f"-d{d}",
            f"-a{a}",
            "-i250",
        ]
        print(f"Running: {nvargs}")
        try:
            result = subprocess.run(
                nvargs,
                capture_output=True,
                text=True,
                cwd=dir,
                # check=True
            )
            int_bh = 0 if a != "bh" else 1
            int_gpu = 0 if d != "gpu" else 1
            print(f"Output:\n{result.stderr}")
            filter_and_save_nvprof(result.stderr, f"benchmarks/nv_N{int(n)}_BH{int_bh}_GPU{int_gpu}.csv")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred for args: {nvargs}")
            print(f"Error message: {e.stderr}")
