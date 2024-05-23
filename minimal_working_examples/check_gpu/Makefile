
check_gpu_clang: check_gpu.c
	clang -fopenmp --offload-arch=sm_89 -o check_gpu check_gpu.c

check_gpu_gcc: check_gpu.c
	gcc -fopenmp -foffload=nvptx-none -o check_gpu check_gpu.c

.PHONY: clean
clean:
	-rm check_gpu
	-rm *.o
	-rm output.log
	-rm slurm-*
