import matplotlib.pyplot as plt

def compare_gpu_cpu_serial(dataset, savepath, num_teams_gpu=3200, num_threads_gpu=307200, num_threads_cpu=32,
                                    colours={"GPU":"blue", "CPU":"red", "serial":"green"}, plot_type="errorbar",
                                    use_mb=False, bbox_style = dict(boxstyle="round", fc="lightsteelblue",
                                                                    edgecolor="royalblue", linewidth=1.5)):
        
    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(8,4))
    
    for mode in ["GPU", "CPU", "serial"]:
        
        #* Load timing data for each mode into sub-dataset for plotting.
        if mode == "GPU":
            num_teams   = num_teams_gpu
            num_threads = num_threads_gpu
        elif mode == "CPU":
            num_teams   = 1
            num_threads = num_threads_cpu
        else:
            num_teams   = 1
            num_threads = 1
    
        mode_data = dataset.data.loc[(dataset.data["Number of teams"] == num_teams) &
                                    (dataset.data["Number of threads"] == num_threads) &
                                    (dataset.data["Mode"] == mode)]
        
        #* Plot results on axis depending on primordial chemistry setting
        for ax, chem in zip(axs.ravel()[:2], [1,2]):
            
            _data = mode_data.loc[mode_data["Primordial chemistry"] == chem]
            
            mean_time  = _data.loc[:, "Mean time (s)"].to_numpy()
            stdev      = _data.loc[:, "Standard deviation (s)"].to_numpy()
            field_size = _data.loc[:, "Field size (gb)"].to_numpy()
            
            if use_mb:
                field_size = field_size * 1024
            
            label = mode if chem == 1 else None
            
            if plot_type == "fill_between":
                upper = mean_time + stdev
                lower = mean_time - stdev
                ax.fill_between(field_size, lower, upper, alpha=0.5,
                                color=colours[mode], label=label)
            elif plot_type == "errorbar":
                ax.errorbar(field_size, mean_time, yerr=stdev,
                            color=colours[mode], label=label,
                            capsize=4., marker="o", markersize=3.,
                            linewidth=0, elinewidth=0.5)
                ax.plot(field_size, mean_time, marker=None, linestyle="dashed",
                        alpha=0.7, color=colours[mode])
            else:
                print("plot_type `%s` not recognised.\nPlease try again..." % plot_type)
                exit()
    
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
        
        ax.tick_params(axis="both", which="both", direction="inout")
        
        ax.grid(alpha=0.8)
        
        if chem == 1:
            ax.legend(loc="upper right",
                      #bbox_to_anchor=(0.75, 0.9, 0.23, 0.1),
                      title="Parallelism")
            
    #* Add table to plot to show settings of the parallelism
    table_rows    = ["GPU", "CPU", "Serial"]
    table_columns = ["# Teams", "# Threads"]
    table_content = [[num_teams_gpu, num_threads_gpu], [1, num_threads_cpu], [1, 1]]
    colour               = "lavender" #64b7c9"
    table_width          = 0.5
    table_cell_colours   = [[colour]*len(i) for i in table_content]
    table_row_colours    = [colour] * len(table_rows)
    table_column_colours = [colour] * len(table_columns)
    # axs.ravel()[-1].set_axis_off()
    axs.ravel()[-1].table(cellText=table_content, rowLabels=table_rows,
                            colLabels=table_columns, loc="lower right",
                            colWidths=[table_width/len(table_columns)]*len(table_columns),
                            cellColours=table_cell_colours,
                            rowColours=table_row_colours,
                            colColours=table_column_colours,
                            zorder=100)
            
    fig.tight_layout()
    if "." not in savepath:
        savepath += ".png"
    elif savepath.split(".")[1] != "png":
        savepath = savepath.split(".")[0] + ".png"
    
    fig.subplots_adjust(wspace=0.)
    fig.savefig(savepath, bbox_inches="tight")