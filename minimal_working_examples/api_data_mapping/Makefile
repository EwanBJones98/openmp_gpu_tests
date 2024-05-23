omp_flags = -fopenmp --offload-arch=sm_89

all: data_map.c
	clang $(omp_flags) -lm -g -o data_map data_map.c

.PHONY: clean
clean:
	-rm slurm-*
	-rm data_map
	-rm *.o
	-rm output.log