omp_flags = -fopenmp --offload-arch=sm_89

time_data_transfer: time_data_transfer.c
	clang $(omp_flags) -lm -o time_data_transfer time_data_transfer.c

.PHONY: clean
clean:
	-rm slurm-*
	-rm time_data_transfer
	-rm *.o
	-rm output.log