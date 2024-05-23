omp_flags = -fopenmp --offload-arch=sm_89

all: map_struct.c
	clang $(omp_flags) -lm -g -o map_struct map_struct.c

.PHONY: clean
clean:
	-rm slurm-*
	-rm map_struct
	-rm *.o
	-rm output.log