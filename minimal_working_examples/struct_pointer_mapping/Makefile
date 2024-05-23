omp_flags = -fopenmp --offload-arch=sm_89

struct_mapping: struct_mapping.c
	clang $(omp_flags) -lm -g -o struct_mapping struct_mapping.c

.PHONY: clean
clean:
	-rm slurm-*
	-rm struct_mapping
	-rm *.o
	-rm output.log