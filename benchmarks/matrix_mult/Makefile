omp_flags = -fopenmp --offload-arch=sm_89

matrix_mult: matrix_mult.c
	clang $(omp_flags) -lm -g -o matrix_mult matrix_mult.c

.PHONY: clean
clean:
	-rm slurm-*
	-rm matrix_mult
	-rm *.o
	-rm output.log