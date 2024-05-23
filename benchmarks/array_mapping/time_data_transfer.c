#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

void time_gpu(double *array, int array_length, int num_iter, double *transfer_time)
{
    double start_time;
    *transfer_time = 0;

    for (int iter=0; iter<num_iter; iter++)
    {
        start_time = omp_get_wtime();
        #pragma omp target enter data map(to:array[:array_length])
        *transfer_time += (omp_get_wtime() - start_time) / num_iter;

        #pragma omp target exit data map(delete:array[:array_length])
    }
}


int main(int argc, char *argv[])
{

    int array_length;
    double *array, array_memory;

    array_length = (int) 1e6;
    array_memory = sizeof(double) * array_length;
    array = malloc(array_memory);

    int num_iter = 1;
    double time;
    time_gpu(array, array_length, num_iter, &time);

    fprintf(stdout, "------ Results ------\n");
    fprintf(stdout, "  Time of data transfer = %g s\n", time);
    fprintf(stdout, "  Size of array = %g MB\n", array_memory/1e6);
    fprintf(stdout, "  Rate of data transfer = %g s/MB\n", time * 1e6 / array_memory);

    return 1;
}