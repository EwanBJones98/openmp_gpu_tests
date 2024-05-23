omp_flags = -fopenmp --offload-arch=sm_89

all: custom_mapping_interface.c
	clang $(omp_flags) -lm -g -o custom_mapping custom_mapping_interface.c

.PHONY: clean
clean:
	-rm slurm-*
	-rm custom_mapping
	-rm *.o
	-rm output.log