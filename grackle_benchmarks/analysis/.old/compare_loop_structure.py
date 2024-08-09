import matplotlib.pyplot as plt

def compare_loop_structure(dataset, savepath, use_mb=True,
                      num_teams_gpu=3200, num_threads_gpu=409600,
                      bbox_style = dict(boxstyle='round', fc='orange',
                                        edgecolor="black", linewidth=2.)):
    print("*** DATASET ***")
    print(dataset.data.to_string())
    
    gpu_data = dataset.data.loc[(dataset.data["Number of teams"] == num_teams_gpu) &
                            (dataset.data["Number of threads"] == num_threads_gpu) & 
                            (dataset.data["Mode"] == "GPU")]
    
    print("*** GPU DATA ***")
    print(gpu_data.to_string())
    
    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(8,4))
    
    for chem_ind, chem in enumerate([1,2]):
        ax  = axs.ravel()[chem_ind]
        
        _data = gpu_data.loc[gpu_data["Primordial chemistry"] == chem]
        
        print("*** _DATA ***")
        print(_data.to_string())
        
        one_loop_data = {"mean":_data.loc[:, "Mean time one loop (s)"].to_numpy(),
                         "stdev":_data.loc[:, "Standard deviation one loop (s)"].to_numpy(),
                         "field size":_data.loc[:, "Field size (gb)"].to_numpy()}
        
        two_loop_data = {"mean":_data.loc[:, "Mean time two loops (s)"].to_numpy(),
                         "stdev":_data.loc[:, "Standard deviation two loops (s)"].to_numpy(),
                         "field size":_data.loc[:, "Field size (gb)"].to_numpy()}
            
        colours = ["red", "blue"]
        labels  = ["one loop", "two loops"]
        
        for dset_ind, dset in enumerate([one_loop_data, two_loop_data]):
            
            mean_time  = dset["mean"]
            stdev      = dset["stdev"]
            field_size = dset["field size"]
            
            if use_mb:
                field_size = field_size * 1024
            
            ax.errorbar(field_size, mean_time, yerr=stdev,
                            color=colours[dset_ind], label=labels[dset_ind],
                            capsize=4., marker="o", markersize=3.,
                            linewidth=0, elinewidth=0.5)
            ax.plot(field_size, mean_time, marker=None, linestyle="dashed",
                        alpha=0.7, color=colours[dset_ind])
    
    #* Configure plot
    for ax, chem in zip(axs.ravel()[:2], [1,2]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        if chem == 1:
            ax.set_ylabel("Mean calculation time (s)")
            
        if use_mb:
            ax.set_xlabel("Field size (mb)")
        else:
            ax.set_xlabel("Field size (gb)")
        
        ax.text(0.05, 0.95, "Primordial chemistry = %d" % chem, transform=ax.transAxes,
                bbox=bbox_style, verticalalignment="top",
                horizontalalignment="left")
        
        if chem == 1:
            ax.legend(loc="best", bbox_to_anchor=(0.8, 0.9, 0.15, 0.1))
            
    #* Add table to plot to show settings of the parallelism
    table_rows    = ["GPU"]
    table_columns = ["# Teams", "# Threads"]
    table_content = [[num_teams_gpu, num_threads_gpu]]
    colour               = "#64b7c9"
    table_width          = 0.5
    table_cell_colours   = [[colour]*len(i) for i in table_content]
    table_row_colours    = [colour] * len(table_rows)
    table_column_colours = [colour] * len(table_columns)
    axs.ravel()[-1].table(cellText=table_content, rowLabels=table_rows,
                            colLabels=table_columns, loc="lower right",
                            colWidths=[table_width/len(table_columns)]*len(table_columns),
                            cellColours=table_cell_colours,
                            rowColours=table_row_colours,
                            colColours=table_column_colours)
            
    fig.tight_layout()
    if "." not in savepath:
        savepath += ".png"
    elif savepath.split(".")[1] != "png":
        savepath = savepath.split(".")[0] + ".png"
    
    fig.subplots_adjust(wspace=0.)
    fig.savefig(savepath, bbox_inches="tight")