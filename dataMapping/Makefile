CC = gcc

GPU_FLAGS = -foffload=nvptx-none -fcf-protection=none -no-pie
EXTRA_GPU_FLAGS = -foffload=-misa=sm_75 -foffload=-lm -fno-stack-protector

structTest: structTest.c
	$(CC) -o structTest -fopenmp $(GPU_FLAGS) structTest.c -lm

.PHONY: clean

arrayTest: arrayTest.c
	$(CC) -o arrayTest -fopenmp $(GPU_FLAGS) arrayTest.c -lm
clean:
	-rm structTest
	-rm arrayTest
	-rm *.o
	-rm slurm*
	-rm output_structTest.log
	-rm output_arrayTest.log