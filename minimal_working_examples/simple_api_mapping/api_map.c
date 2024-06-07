#include <stdlib.h>
#include <stdio.h>
#ifdef _OPENMP
    #include <omp.h>
#endif

typedef struct
{
    int n;
    double *arr;
} vector;

int main(int argc, char *argv[])
{

    int device = omp_get_default_device();
    int host   = omp_get_initial_device();

    vector my_vector;
    my_vector.n = 100;
    my_vector.arr = malloc(sizeof(double) * my_vector.n);
    for (int i=0; i<my_vector.n; i++)
    {
        my_vector.arr[i] = -1;
    }
    
    double *arr_dev;
    arr_dev = omp_target_alloc(sizeof(double) * my_vector.n, device);
    omp_target_memcpy(arr_dev, my_vector.arr, sizeof(double) * my_vector.n, 0, 0, device, host);

    #pragma omp target teams distribute parallel for is_device_ptr(arr_dev)
    for (int i=0; i<my_vector.n; i++)
    {
        arr_dev[i] = 44;
    }

    omp_target_memcpy(my_vector.arr, arr_dev, sizeof(double)*my_vector.n, 0, 0, host, device);

    fprintf(stdout, "Array[0] = %g\n", my_vector.arr[0]);

    return 0;
}