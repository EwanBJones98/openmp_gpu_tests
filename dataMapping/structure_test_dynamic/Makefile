CC = gcc

GPU_FLAGS = -foffload=nvptx-none -fcf-protection=none -no-pie
EXTRA_GPU_FLAGS = -foffload=-misa=sm_75 -foffload=-lm -fno-stack-protector

map_struct: map_struct.c
	$(CC) -o map_struct -fopenmp $(GPU_FLAGS) map_struct.c -lm -Wall -g

.PHONY: clean
clean:
	-rm map_struct
	-rm *.o
	-rm slurm*
	-rm output_map_struct.log