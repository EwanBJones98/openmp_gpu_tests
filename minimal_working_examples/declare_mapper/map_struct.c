#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef _OPENMP
    #include <omp.h>
#endif


typedef struct
{
    double *values;
    int length;
} vector;


#pragma omp declare mapper(vector v) map(v.length, v.values[:v.length])


int main(int argc, char *argv[])
{

    vector my_vector;
    int length = 1000;

    my_vector.length       = length;
    my_vector.values       = malloc(my_vector.length * sizeof(double));

    for (int i=0; i<my_vector.length; i++)
    {
        my_vector.values[i] = -1;
    }

    #pragma omp target enter data map(to:my_vector)

    #pragma omp target teams distribute parallel for
    for (int i=0; i<my_vector.length; i++)
    {
        my_vector.values[i] = 44;
    }

    #pragma omp target exit data map(from:my_vector)

    if (fabs(my_vector.values[0] - 44) <= 1e-10)
    {
        fprintf(stdout, "map_struct: PASSED\n");
    } else {
        fprintf(stdout, "map_struct: FAILED\n");
    }

    my_vector.length = 0;
    free(my_vector.values);
    my_vector.values = NULL;

    return 0;
}