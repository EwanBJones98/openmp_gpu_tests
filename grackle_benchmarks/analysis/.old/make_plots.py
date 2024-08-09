from Timings import TimingsData
from compare_gpu_cpu_serial import compare_gpu_cpu_serial
from compare_loop_structure import compare_loop_structure
from compare_optimisation_levels import compare_optimisation_levels
from worksharing_test import worksharing_test

import os.path as path
import seaborn as sns


sns.set_context("paper")
sns.set_style()


#! >>> Figures to make <<<
make_gpu_cpu_serial  = False
make_loop_structure  = False
make_optimization    = False
make_worksharing     = True
#! >>>>>>>>>><<<<<<<<<<


master_dir   = path.join("/home", "ejones", "codes", "openmp_gpu_tests",
                        "grackle_benchmarks")
timings_dir  = path.join(master_dir, "timings")
analysis_dir = path.join(master_dir, "analysis")


# >>> Plot GPU vs CPU vs serial performance <<<
timings_path = str(path.join(timings_dir,  "gpu_cpu_serial.txt"))
save_path    = str(path.join(analysis_dir, "compare_gpu_cpu_serial.png"))

if make_gpu_cpu_serial:
    timings      = TimingsData(timings_path, "gpu_cpu_serial_timings")
    compare_gpu_cpu_serial(timings, save_path, num_teams_gpu=3200, num_threads_gpu=409600,
                            num_threads_cpu=32, use_mb=True)


# >>> Plot number of loops' effect on performance <<<
filepath_loop = str(path.join(timings_dir, "loop_structure_test.txt"))
save_path     = str(path.join(analysis_dir, "loop_structure.png"))

if make_loop_structure:
    timings_loop  = TimingsData(filepath_loop, "loop structure",
                                loop_structure_data=True)
    compare_loop_structure(timings_loop, save_path, num_teams_gpu=3200, num_threads_gpu=409600)


# >>> Plot optimization level's effect on performance <<<
filepath_optimisation = str(path.join(timings_dir, "optimization_flag.txt"))
save_path             = str(path.join(analysis_dir, "optimisation_level.png"))

if make_optimization:
    timings_optimization = TimingsData(filepath_optimisation, "optimisation")
    compare_optimisation_levels(timings_optimization, save_path)

# >>> Plot thread/team worksharing balance's effect on performance <<<
filepath_worksharing = str(path.join(timings_dir, "worksharing_test.txt"))
save_path            = str(path.join(analysis_dir, "worksharing_test.png"))

if make_worksharing:
    timings_worksharing  = TimingsData(filepath_worksharing, "worksharing",
                                    optimisation_flag=False)
    worksharing_test(timings_worksharing, save_path, [1000,100,100], 2)