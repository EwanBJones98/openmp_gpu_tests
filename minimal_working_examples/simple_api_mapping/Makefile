omp_flags = -fopenmp --offload-arch=sm_89

api_map: api_map.c
	clang $(omp_flags) -lm -g -o api_map api_map.c

.PHONY: clean
clean:
	-rm slurm-*
	-rm api_map
	-rm *.o
	-rm output.log
