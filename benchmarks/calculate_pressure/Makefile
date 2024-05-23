CC = gcc

#GPU_OMPFLAGS = -foffload=nvptx-none -foffload=-misa=sm_30 -foffload=-lm -fcf-protection=none -fno-stack-protector
GPU_OMPFLAGS = -foffload=nvptx-none -fcf-protection=none -no-pie

testGPU: calculate_pressure.c
	$(CC) -D GPU -o calculate_pressure -fopenmp $(GPU_OMPFLAGS) calculate_pressure.c -lm

testCPU: calculate_pressure.c
	$(CC) -D CPU -o calculate_pressure -fopenmp calculate_pressure.c

testSerial: calculate_pressure.c
	$(CC) -D SERIAL -o calculate_pressure calculate_pressure.c

.PHONY: clean

clean:
	-rm calculate_pressure
	-rm *.o
	-rm output.log
	-rm slurm-*