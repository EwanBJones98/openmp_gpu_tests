#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

/*
    This is a remake of the example shown at https://passlab.github.io/Examples/contents/Chap_devices/11_target_enter_data_and_target_exit_data_Constructs.html
*/


typedef struct
{
    double *values;
    int length;
} vector;


void init_vector(vector *vec, int length)
{
    vec->length = length;
    vec->values = malloc(vec->length * sizeof(double));

    #pragma omp target enter data map(alloc:vec->length, vec->values[:vec->length])
}

void free_vector(vector *vec)
{
    #pragma omp target exit data map(delete:vec->length, vec->values[:vec->length])

    vec->length = 0;
    free(vec->values);
    vec->values = NULL;
}

void set_vector(vector *vec, double value)
{
    for (int i=0; i<vec->length; i++)
    {
        vec->values[i] = value;
    }
}

void update_on_device(vector *vec, double value)
{
    #pragma omp target update to(vec->length, vec->values[:vec->length])

    #pragma omp target teams distribute parallel for map(to:value)
    for (int i=0; i<vec->length; i++)
    {
        vec->values[i] = value;
    }

    #pragma omp target update from(vec->values[:vec->length])
}

int check_array(double *input_array, int array_length, double expected_value)
{
    for (int i=0; i<array_length; i++)
    {
        if (fabs(input_array[i] - expected_value) > 1e-10)
        {
            return 0;
        }
    }
    return 1;
}


int main(int argc, char *argv[])
{

    vector my_vector;
    int length = 1000;

    init_vector(&my_vector, length);

    set_vector(&my_vector, -1);

    update_on_device(&my_vector, 44);

    if (check_array(my_vector.values, my_vector.length, 44))
    {
        fprintf(stdout, "test PASSED\n");
    } else {
        fprintf(stdout, "test FAILED\n");
    }

    free_vector(&my_vector);

    return 0;
}