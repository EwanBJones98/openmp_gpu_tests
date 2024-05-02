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

    double pressure[5];
    double density[5];
} fields_static;

void initialise_fields(fields_static *my_fields, double initial_value)
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
    fields_static my_fields;
    my_fields.field_length = 5;
    // allocate_memory_fields(&my_fields, 5);
    initialise_fields(&my_fields, 1e-20);

    fprintf(stdout, "Initial density:\n");
    print_array(my_fields.density, my_fields.field_length);
    fprintf(stdout, "Initial pressure:\n");
    print_array(my_fields.pressure, my_fields.field_length);

    #pragma omp target enter data map(to: my_fields)

    #pragma omp target teams distribute parallel for
    for (int i=0; i < my_fields.field_length; i++){
        my_fields.density[i]  = 1.;
        my_fields.pressure[i] = 1.;
    }

    #pragma omp target exit data map(from: my_fields)

    fprintf(stdout, "Final density:\n");
    print_array(my_fields.density, my_fields.field_length);
    fprintf(stdout, "Final pressure:\n");
    print_array(my_fields.pressure, my_fields.field_length);

    return 1;
}