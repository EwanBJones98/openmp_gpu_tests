import matplotlib.pyplot as plt
from numpy import argsort
from matplotlib.ticker import MultipleLocator

def compare_optimisation_levels(dataset, savepath, use_mb=False,
                                field_dimensions=[1000,100,100],
                                colours={"GPU":"blue", "CPU":"red", "serial":"green"},
                                bbox_style = dict(boxstyle='round', fc='orange',
                                                    edgecolor="black", linewidth=2.,
                                                    alpha=0.7)):
    
    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(8,4))
    
    for ax, chem in zip(axs, [1,2]):
    
        for mode in ["GPU", "CPU", "serial"]:
                        
            data = dataset.data.loc[(dataset.data["Mode"] == mode) &
                                    (dataset.data["Primordial chemistry"] == chem) &
                                    (dataset.data["Fields i dimension"] == field_dimensions[0]) &
                                    (dataset.data["Fields j dimension"] == field_dimensions[1]) &
                                    (dataset.data["Fields k dimension"] == field_dimensions[2])]
            
            xsort = argsort(data.loc[:, "Optimisation flag"].to_numpy())
            
            opt_level = data.loc[:, "Optimisation flag"].to_numpy()[xsort]
            mean_time = data.loc[:, "Mean time (s)"].to_numpy()[xsort]
            stdev     = data.loc[:, "Standard deviation (s)"].to_numpy()[xsort]
        
            if use_mb:
                field_size = field_size * 1024
                
            lab = mode if chem == 1 else None
            
            ax.errorbar(opt_level, mean_time, yerr=stdev, color=colours[mode],
                        capsize=4., marker="o", markersize=3., linewidth=0,
                        elinewidth=0.5, label=lab)
            ax.plot(opt_level, mean_time, marker=None, linestyle="dashed",
                    alpha=0.7, color=colours[mode])
                
    #* Configure plot
    for ax, chem in zip(axs.ravel()[:2], [1,2]):
        ax.set_yscale("log")
    
        ax.set_xlabel("Optimisation level")
        
        ax.xaxis.set_major_locator(MultipleLocator(1))
        
        ax.text(0.85, 0.95, "Primordial chemistry = %d" % chem, transform=ax.transAxes,
                bbox=bbox_style, verticalalignment="top", horizontalalignment="right")
        
        if chem == 1:
            ax.set_ylabel("Mean calculation time (s)")
            
            ax.legend(loc="best", bbox_to_anchor=(0.1, 0.9, 0.15, 0.1))    
            
        if chem == 2:
            ax.text(0.85, 0.05, f"Field dimensions = {field_dimensions}", transform=ax.transAxes,
                    bbox=bbox_style, verticalalignment="bottom", horizontalalignment="right")
            
    fig.suptitle("See data table for number of teams/threads as it depends on the optimisation level.")
                        
    fig.tight_layout()
    if "." not in savepath:
        savepath += ".png"
    elif savepath.split(".")[1] != "png":
        savepath = savepath.split(".")[0] + ".png"
    
    fig.subplots_adjust(wspace=0.)
    fig.savefig(savepath, bbox_inches="tight")
                
                
                
            