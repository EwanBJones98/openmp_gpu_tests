CC = clang

GPU_FLAGS = -march=nvptx

declare_mapper_test: declare_mapper_test.c
	$(CC) -o declare_mapper_test -fopenmp $(GPU_FLAGS) declare_mapper_test.c -lm

.PHONY: clean
clean:
	-rm declare_mapper_test
	-rm *.o
	-rm slurm*
	-rm output_declare_mapper_test.log