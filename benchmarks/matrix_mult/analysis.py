import os.path as p
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_benchmark(filepath: str):
    lambda_skip_row = lambda x: x in [1]

    col_names = ["matrix_dim", "num_iterations", "num_teams", "num_threads",
                 "num_threads_per_team", "total_time", "transfer_time", "calculation_time"]
    
    dtypes = {key:np.int64 for key in col_names}
    for time_type in ["total", "transfer", "calculation"]:
        dtypes[f"{time_type}_time"] = np.float64
    
    return pd.read_csv(filepath, sep="|", dtype=dtypes, header=0, names=col_names,
                    skiprows=lambda_skip_row, skipinitialspace=True)
   
    
def read_multi_benchmark(filenames):
    dataframes = [read_benchmark(fname) for fname in filenames]
    return pd.concat(dataframes)

def plot_timing_vs_matrix_size(df, nteams_bin_edges):
    
    nteams_bin_centers = (nteams_bin_edges[1:] + nteams_bin_edges[:-1]) / 2
    
    fig, axs = fig.subplots(nrows=np.ceil(nteams_bin_centers/3), ncols=3, figsize=(10,8))
    
    for i, (lower_edge, upper_edge) in enumerate(zip(nteams_bin_edges[:-1], nteams_bin_edges[1:])):
        ax = axs.ravel()[i]
        
        df_binned = df.loc[(df["num_teams"] >= lower_edge) & (df["num_teams"] < upper_edge)]
        
        
        for (dset, dlab) in zip(["transfer_time", "calculation_time", "total_time"],
                                ["data transfer", "calculation", "data transfer + calculation"]):
            ax.plot(df_binned[:,"matrix_dim"], df_binned[:,dset], linewidth=0., label=dlab)
            
        
        ax.yscale("log")
        ax.xscale("log")
        ax.set_xlabel("Square Matrix Side Length")
        ax.set_ylabel("Average Elapsed Time (s)")
        
        if i == 0:
            ax.legend(title="Timing Interval")
        
    fig.tight_layout()
    fig.savefig("/home/ejones/codes/openmp_gpu_tests/benchmarks/matrix_mult/timing_vs_matrix_dimensions.png", bbox_inches="tight")
    
    
if __name__ == "__main__":

    sns.set_style("darkgrid")

    filedir   = "/home/ejones/codes/openmp_gpu_tests/benchmarks/matrix_mult/Datasets/"
    filenames = [str(p.join(p.dirname(filedir), f"benchmark_{id}.txt"))\
                    for id in ["dim100","table1","table2","teams1"]]

    df = read_multi_benchmark(filenames)
    
    
    num_teams_unique   = sorted(df.loc[:,"num_teams"].unique())
    matrix_dim_unique  = sorted(df.loc[:,"matrix_dim"].unique())
    
    