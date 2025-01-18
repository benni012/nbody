import itertools
import subprocess

n_values = [
    10, 20, 30, 40, 50, 60, 70, 80, 90, 
    100, 200, 300, 400, 500, 600, 700, 800, 900, 
    1000, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 
    1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 
    1e5
]
d_values = ["gpu", "cpu"]
a_values = ["bh"]
i_value = "1e3"  # Fixed value

cpp_executable = "./nbody"

combinations = itertools.product(n_values, d_values, a_values)

for n, d, a in combinations:
    args = [
        cpp_executable,
        f"-n{int(n)}",
        f"-d{d}",
        f"-a{a}",
        f"-i{i_value}",
        "-b",
    ]
    
    print(f"Running: {' '.join(args)}")
    
    try:
        result = subprocess.run(
            args,
            # capture_output=False,
            # text=True,
            # check=True
        )
        print(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred for args: {args}")
        print(f"Error message: {e.stderr}")
