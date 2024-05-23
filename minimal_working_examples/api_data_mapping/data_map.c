#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
    #include <omp.h>
#endif


typedef struct
{
    int length;
    double *values;
} vector;



int main(int argc, char *argv[])
{
    vector my_vector;
    my_vector.length = 10;
    my_vector.values = malloc(sizeof(double) * my_vector.length);

    int target_device_num = omp_get_default_device();
    int host_device_num   = omp_get_initial_device();
    size_t array_size = sizeof(double) * my_vector.length;

    double *target_ptr = omp_target_alloc(array_size, target_device_num);
    omp_target_memcpy(target_ptr, my_vector.values, array_size, 0, 0, target_device_num, host_device_num);
    omp_target_associate_ptr(my_vector.values, target_ptr, array_size, 0, target_device_num);

    #pragma omp target teams distribute parallel for
    {
        for (int i=0; i<my_vector.length; i++)
        {
            my_vector.values[i] = 3;
        }
    }

    #pragma omp target update from(my_vector.values[:my_vector.length])

    // omp_target_memcpy(my_vector.values, target_ptr, array_size, 0, 0, host_device_num, target_device_num);

    fprintf(stdout, "%f", my_vector.values[0]);


    return 1;
}