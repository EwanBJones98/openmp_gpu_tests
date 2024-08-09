import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

from read_data import read_timing_data

sns.set_style()
sns.set_context("paper")


if __name__ == "__main__":
    
    # Location to save the figure to
    save_dir      = os.path.join("/home", "ejones", "codes", "openmp_gpu_tests", "grackle_benchmarks",
                                    "analysis", "plots")
    save_plotname = "optimisation_level.pdf"
    save_plotpath = os.path.join(save_dir, save_plotname)

    # The location of the timing file on disk
    data_dir      = os.path.join("/home", "ejones", "codes", "openmp_gpu_tests", "grackle_benchmarks",
                                    "timings", "optimised_worksharing")
    data_namebase = "optimisationLevel_"
    
    # Define how the timing file is structured
    data_columns  = ["Mode", "Primordial chemistry", "Field dimensions i", "Field dimensions j", "Field dimensions k",
                        "Number of timing iterations", "Number of teams", "Number of threads per team", "Number of threads",
                        "Calculation time", "Standard deviation calculation time", "Data transfer time"]
    column_dtypes = {colname:"int" for colname in data_columns}
    
    column_dtypes["Mode"]                                = "str"
    column_dtypes["Calculation time"]                    = "float"
    column_dtypes["Standard deviation calculation time"] = "float"
    column_dtypes["Data transfer time"]                  = "float"
    
    # Read in data from all timing files
    all_data = {}
    for opt_lvl in [0,1,2]:
        
        data_filepath = os.path.join(data_dir, f"{data_namebase}{opt_lvl}.txt")
        dataset = read_timing_data(data_filepath, data_columns, column_dtypes)
        
        all_data[opt_lvl] = pd.DataFrame.from_dict(dataset)
        
    # Create figure
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex="all", sharey="row", figsize=(9,6))
            
    # Get number of unique threads per team used throughout all optimisation levels
    #  and create colourmap
    all_threadsPerTeam = np.concatenate([all_data[opt_lvl]["Number of threads per team"][:] \
                                            for opt_lvl in [0,1,2]])
    unique_threadsPerTeam = np.sort(pd.unique(all_threadsPerTeam))
    cm_subsection = np.linspace(0., 1., len(unique_threadsPerTeam))
    colours = {}
    for i, val in enumerate(unique_threadsPerTeam):
        colours[val] = cm.turbo(cm_subsection[i])
    
    for opt_lvl in [0,1,2]:
        data = all_data[opt_lvl]
        
        for threadsPerTeam in unique_threadsPerTeam:
            
            data_subset = data.loc[data["Number of threads per team"] == threadsPerTeam]
            
            fieldsize = 1
            for unitvec in ["i", "j", "k"]:
                fieldsize *= data_subset[f"Field dimensions {unitvec}"][:]
                
            calctime_upper = data_subset["Calculation time"][:] + data_subset["Standard deviation calculation time"][:]
            calctime_lower = calctime_upper - 2*data_subset["Standard deviation calculation time"][:]
                
            axs[0,opt_lvl].plot(fieldsize, data_subset["Calculation time"][:], color=colours[threadsPerTeam], alpha=0.7)
            # axs[0,opt_lvl].fill_between(fieldsize, calctime_lower, calctime_upper, color=colours[threadsPerTeam], alpha=0.3)
            
            axs[1,opt_lvl].plot(fieldsize, data_subset["Data transfer time"][:], color=colours[threadsPerTeam], alpha=0.7)
            
    # Add legends to axis
    # dummy_lines = []
    # for opt_lvl, style in zip([0,1,2], ["solid", "dashed", "dotted"]):
    #     dummy_lines.append(axs[0].plot([], [], color="black", linestyle=style, markersize=0., label=opt_lvl)[0])
    # style_legend = axs[0].legend(handles=dummy_lines, title="Optimisation level", loc="upper left")
    # axs[0].add_artist(style_legend)
    
    dummy_lines = []
    for threadsPerTeam in np.sort(unique_threadsPerTeam):
        dummy_lines.append(axs[0,0].plot([], [], color=colours[threadsPerTeam], linestyle="solid", markersize=0., label=threadsPerTeam)[0])
    colour_legend = axs[0,0].legend(handles=dummy_lines, title="Number of threads per team", loc="lower right", fontsize=8)
    axs[0,0].add_artist(colour_legend)
    
    # Axis configuration
    for opt_lvl, ax in enumerate(axs[0,:]): ax.set_title(f"Optimisation Level {opt_lvl}")
    for ax in axs[1,:]: ax.set_xlabel("Number of Particles")
    axs[0,0].set_ylabel("Mean Calculation Time (s)")
    axs[0,0].set_xscale("log")
    axs[0,0].set_yscale("log")
    
    axs[1,0].set_ylabel("Data Transfer Time (s)")
    axs[1,0].set_yscale("log")
    
    for ax in axs.ravel(): ax.tick_params(which="both", direction="inout", axis="both")
    for ax in axs.ravel(): ax.grid(alpha=0.4)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(save_plotpath, bbox_inches="tight")