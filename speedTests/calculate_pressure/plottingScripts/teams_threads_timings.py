import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

sns.set_style()

def load_timing_data(timings_path: str = "default") -> pd.DataFrame:

    if timings_path == "default":
        timings_path = os.path.join("/home", "ejones", "codes", "openmp_gpu_tests",
                                        "speedTests", "calculate_pressure", "timings.txt")
        
    if not os.path.isfile(timings_path):
        print(f"File 'timings.txt' not found at {str(timings_path)}")
        print("\nExiting...")
        exit()
       
    data = defaultdict(lambda: [])
    with open(str(timings_path), "r") as fptr:
        for line_i, line in enumerate(fptr.readlines()):
            
            # Read names of columns into a list
            if line_i == 0:
                column_names = line.split("|")[1:-1]
                column_names = [n.strip() for n in column_names]
                continue
            
            # Ignore empty lines
            if line in ["\n", "\r\n"]:
                continue
            
            # Store data on this line into the dictionary
            line_data = line.split("|")[1:-1]
            for value_i, value in enumerate(line_data):
                value = value.strip()
                if value_i != 0:
                    value = float(value)
                data[column_names[value_i]].append(value)
                
    return pd.DataFrame(data)

def plot_calculation_time(data: pd.DataFrame) -> None:

    data_sorted = data.sort_values("# teams")
    
    data_sorted["average mapping time (s)"] = data_sorted.loc[:,"average total time (s)"]\
                                            - data_sorted.loc[:,"average calculation time (s)"]
                                            
    column_rename_mapping = {"average total time (s)": "calculation + data mapping",
                            "average calculation time (s)": "calculation",
                            "average mapping time (s)": "data mapping"}
    
    data_sorted.rename(columns=column_rename_mapping, inplace=True)
    
    data_melted = data_sorted.melt(value_vars=["calculation", "data mapping", "calculation + data mapping"],
                                    value_name="time", var_name="Benchmark", id_vars=["# threads per team", "# teams"])
    
    # Define mapping between number of teams and threads
    num_threads_per_team = data_melted["# threads per team"]
    def num_teams_to_threads(num_teams):
        return num_teams * num_threads_per_team
    def num_threads_to_teams(num_threads):
        return num_threads / num_threads_per_team
    
    fig, ax_teams = plt.subplots(figsize=(8,6))
    
    # sns.lineplot(x="# teams", y="calculation", data=data_sorted, ax=ax_teams)
    # sns.lineplot(x="# teams", y="data mapping", data=data_sorted, ax=ax_teams)
    # sns.lineplot(x="# teams", y="calculation + data mapping", data=data_sorted, ax=ax_teams)
    
    sns.lineplot(x="# teams", y="time", hue="Benchmark", data=data_melted, ax=ax_teams)
    
    ax_teams.set_xlabel("Number of Teams")
    ax_teams.set_ylabel("Time (s)")
    ax_teams.grid()
    
    # Add secondary x-axis
    ax_threads = ax_teams.secondary_xaxis("top", functions=(num_teams_to_threads, num_threads_to_teams))
    ax_threads.set_xlabel("Number of Threads")
    
    fig.tight_layout()
    fig.savefig("../plots/teams_threads_timings.png", bbox_inches="tight")

if __name__ == "__main__":
    
    data = load_timing_data()

    plot_calculation_time(data)