import re
import pandas as pd
import os


def get_benchmark_df(directory=".") -> pd.DataFrame:
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith("benchmark_results"):
            # Extract N, BH, and GPU information from the filename using regex
            match = re.search(r"N(\d+)_BH(\d+)_GPU(\d+)", filename)
            if match:
                N = int(match.group(1))
                BH = bool(int(match.group(2)))
                GPU = bool(int(match.group(3)))

            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                data.append(
                    {"N": N, "BH": BH, "GPU": GPU, "Function": row["Function"], "Mean Time (us)": row["Mean Time (us)"]}
                )

    return pd.DataFrame(data)


def split_number_and_unit(value):
    import re

    match = re.match(r"([\d.]+)([a-zA-Z/%]*)", str(value))
    if match:
        return float(match.group(1)), match.group(2)
    else:
        return value, None  # If no unit found

conversion_factors = {
    "B/s": 1 / (1024 * 1024),  # Bytes per second to MB/s
    "KB/s": 1 / 1024,          # Kilobytes per second to MB/s
    "GB/s": 1024               # Gigabytes per second to MB/s
}

# Function to scale values to MB/s
def scale_to_mbps(row, unit_col, value_col):
    unit = row[unit_col]
    value = row[value_col]
    if pd.notnull(value) and pd.notnull(unit) and unit in conversion_factors:
        return float(value) * conversion_factors[unit]
    return value

def get_nv_profs(directory=".") -> pd.DataFrame:
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith("nv_"):
            filename = os.path.basename(filename)
            match = re.search(r"N(\d+)_BH(\d+)_GPU(\d+)", filename)
            if match:
                N = int(match.group(1))
                BH = bool(int(match.group(2)))

            # Read the CSV into a DataFrame
            df = pd.read_csv(filename)

            df["Kernel"] = df["Kernel"].replace(func_map)

            # Split units from Min, Max, Avg columns
            for col in ["Min", "Max", "Avg"]:
                df[f"{col}_Value"], df[f"{col}_Unit"] = zip(*df[col].map(split_number_and_unit))

            # Drop original Min, Max, Avg columns
            df.drop(columns=["Min", "Max", "Avg"], inplace=True)

            # Pivot the metrics
            df['Row_ID'] = df.index
            df_melted = df.melt(
                id_vars=['Row_ID', 'Device', 'Kernel', 'Invocations'], 
                value_vars=['Min_Value', 'Max_Value', 'Avg_Value', 'Min_Unit', 'Max_Unit', 'Avg_Unit'],
                var_name='Metric_Detail',
                value_name='Measurement'
            )
            df_melted = df_melted.merge(
                df[['Row_ID', 'Metric Name', 'Metric Description']], 
                on='Row_ID', 
                how='left'
            )
            df_melted['Detail'] = df_melted['Metric_Detail'].str.split('_').str[0]
            df_melted['Type'] = df_melted['Metric_Detail'].str.split('_').str[-1]
            df_pivoted = df_melted.pivot_table(
                index=['Device', 'Kernel', 'Invocations'],
                columns=['Metric Name', 'Detail', 'Type'],
                values='Measurement',
                aggfunc='first'
            )
            df_pivoted.columns = ['_'.join(col).strip() for col in df_pivoted.columns.values]
            df_final = df_pivoted.reset_index()

            # Scale units to MB/s
            for col in df_final.columns:
                if "throughput" in col and "Value" in col:
                    unit_col = col.replace("Value", "Unit")  # Find corresponding unit column
                    if unit_col in df_final.columns:
                        df_final[col] = df_final.apply(scale_to_mbps, axis=1, unit_col=unit_col, value_col=col)
                        df_final[unit_col] = "MB/s"  # Update the unit to MB/s

            # flop count to float
            for col in df_final.columns:
                if "flop_count" in col and "Value" in col:
                    df_final[col] = df_final[col].astype('float64')

            # Add columns
            df_final["N"] = N
            df_final["BH"] = BH
            # Append to list
            data.append(df_final)

    # Combine all dataframes
    return pd.concat(data, ignore_index=True)


markers = ["o", "s", "D", "v", "x", "^", "<", ">", "p", "*"]  # Markers for different functions

func_map = {"bh_kernel(body_t*, octree_t*)": "AccUpdateBH_GPU", "update_pos_kernel(int, body_t*)": "PosUpdate_GPU"}
# func_map = {"AccUpdate_GPU": "bh_kernel(body_t*, octree_t*)", "PosUpdate_GPU": "update_pos_kernel(int, body_t*)"}
