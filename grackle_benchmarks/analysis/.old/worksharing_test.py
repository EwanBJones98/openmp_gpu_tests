import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def worksharing_test(timings_data, savepath, field_dimensions, primordial_chemistry,
                     bbox_style = dict(boxstyle="round", fc="lightsteelblue",
                                       edgecolor="royalblue", linewidth=1.5)):
    
    data = timings_data.data.loc[(timings_data.data["Fields i dimension"] == field_dimensions[0]) &
                                 (timings_data.data["Fields j dimension"] == field_dimensions[1]) &
                                 (timings_data.data["Fields k dimension"] == field_dimensions[2]) &
                                 (timings_data.data["Primordial chemistry"] == primordial_chemistry) &
                                 (timings_data.data["Mode"] == "GPU")]
    
    nthreads_unique = sorted(data["Number of threads per team"].unique())
    min_threads = min(nthreads_unique)
    max_threads = max(nthreads_unique)
    
    cmap_norm     = mpl.colors.LogNorm(vmin=min_threads, vmax=max_threads)
    cmap_mappable = mpl.cm.ScalarMappable(norm=cmap_norm, cmap=mpl.cm.viridis)
    colours       = {nthread:cmap_mappable.to_rgba(nthread) for nthread in nthreads_unique}
    
    use_colourbar = False
    if len(nthreads_unique) > 5:
            use_colourbar = True
    
    fig, ax = plt.subplots()
    
    for nthreads in nthreads_unique:
        
        _data  = data.loc[data["Number of threads per team"] == nthreads]
    
        sort_filt = np.argsort(_data.loc[:, "Number of teams"].to_numpy())
        nteams = _data.loc[:, "Number of teams"].to_numpy()[sort_filt]
        timing = _data.loc[:, "Mean time (s)"].to_numpy()[sort_filt]
        stdev  = _data.loc[:, "Standard deviation (s)"].to_numpy()[sort_filt]
        
        my_label  = nthreads if not use_colourbar else None
        my_colour = colours[nthreads]
        
        _line = ax.errorbar(nteams, timing, yerr=stdev,
                    color=my_colour, label=my_label,
                    capsize=4., marker="o", markersize=3.,
                    linewidth=0, elinewidth=0.5)
        ax.plot(nteams, timing, marker=None, linestyle="dashed",
                alpha=0.7, color=my_colour)
    
    if use_colourbar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0,
                                0.02, ax.get_position().height])
        cbar = fig.colorbar(cmap_mappable, cax=cbar_ax)
        cbar.set_label("Number of threads per team")
    else:
        ax.legend(title="Number of threads per team", loc="upper right")
    
    ax.text(0.05, 0.05, f"Primordial chemistry = {primordial_chemistry}", transform=ax.transAxes,
            verticalalignment="bottom", horizontalalignment="left", bbox=bbox_style)
    ax.text(0.05, 0.15, f"Field dimensions = {field_dimensions}",
            transform=ax.transAxes, verticalalignment="bottom", horizontalalignment="right",
            bbox=bbox_style)
    
    ax.grid(alpha=0.8)
    ax.tick_params(axis="both", which="both", direction="inout")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_xlabel("Number of teams")
    ax.set_ylabel("Mean calculation time (s)")
    
    fig.savefig(savepath, bbox_inches="tight")