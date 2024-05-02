#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

typedef struct
{
    int field_length;

    double *pressure;
    double *density;
} fields_dynamic;

void initialise_fields(fields_dynamic *my_fields, double initial_value)
{
    for (int i=0; i<my_fields->field_length; i++)
    {
        my_fields->density[i]  = initial_value;
        my_fields->pressure[i] = initial_value;
    }
}

void print_array(double *array, int array_length)
{
    fprintf(stdout, "[");
    for (int i=0; i<array_length; i++)
    {
        fprintf(stdout, "%g", array[i]);

        if (i < array_length - 1)
        {
            fprintf(stdout, ",");
        }
    }
    fprintf(stdout, "]\n");
}


int main(int argc, char *argv[])
{
    fields_dynamic my_fields;
    my_fields.field_length = 5;
    my_fields.density = malloc(sizeof(double) * my_fields.field_length);
    my_fields.pressure = malloc(sizeof(double) * my_fields.field_length);
    initialise_fields(&my_fields, 1e-20);

    fprintf(stdout, "Initial density:\n");
    print_array(my_fields.density, my_fields.field_length);
    fprintf(stdout, "Initial pressure:\n");
    print_array(my_fields.pressure, my_fields.field_length);

    #pragma omp target enter data\
            map(to: my_fields.density, my_fields.pressure, my_fields.field_length,\
                    my_fields.density[:my_fields.field_length], my_fields.pressure[:my_fields.field_length])

    // int host_num, target_num;
    // double array_size;
    // array_size = sizeof(double) * my_fields.field_length;
    // host_num   = omp_get_initial_device();
    // target_num = omp_get_default_device();
    // omp_target_memcpy(my_fields.density, my_fields.density, array_size, 0, 0, target_num, host_num);
    // omp_target_memcpy(my_fields.density, my_fields.density, array_size, 0, 0, target_num, host_num);

    #pragma omp target teams distribute parallel for
    for (int i=0; i < my_fields.field_length; i++){
        my_fields.density[i]  = 1.;
        my_fields.pressure[i] = 1.;
    }

    int N;

    #pragma omp target map(tofrom: N)
    {
        N = my_fields.field_length*2 - 1;
    }

    // omp_target_memcpy(my_fields.density, my_fields.density, array_size, 0, 0, host_num, target_num);
    // omp_target_memcpy(my_fields.pressure, my_fields.pressure, array_size, 0, 0, host_num, target_num);
    
    #pragma omp target exit data map(from: my_fields.density[:my_fields.field_length],\
                                           my_fields.pressure[:my_fields.field_length],\
                                           N)

    fprintf(stdout, "Final density:\n");
    print_array(my_fields.density, my_fields.field_length);
    fprintf(stdout, "Final pressure:\n");
    print_array(my_fields.pressure, my_fields.field_length);
    fprintf(stdout, "%d", N);
    return 1;
}