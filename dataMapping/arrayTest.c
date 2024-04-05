#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "structTest.h"

#if _OPENMP
    #include <omp.h>
#endif


int main(int arvc, char *argv[])
{

    // Check GPU is activated correctly
    int active;
    #pragma omp target map(tofrom:active)
    {
        active = omp_is_initial_device();
    }
    if (active == 0){
        fprintf(stdout, "\n GPU is active \n");
    } else {
        fprintf(stdout, "\n GPU not active! Quitting...");
    }

    // Test passing a simple array to the target device and modifying its values

    double *test_array;
    int N;
    N = 10;
    test_array = (double*) malloc(N * sizeof(double));
    for (int i=0; i<N; i++){
        test_array[i] = i*10.;
    }

    fprintf(stderr, "-- Initial values ---\n");
    for (int i=0; i<N; i++){
        fprintf(stderr, "\t %d = %g\n", i, test_array[i]);
    }

    #pragma omp target enter data map(to:test_array[:N], N)

    #pragma omp target teams distribute parallel for
    for (int i=0; i<N; i++){
        test_array[i] *= 2;
    }

    #pragma omp target exit data map(from:test_array[:N])

    fprintf(stderr, "-- Final values ---\n");
    for (int i=0; i<N; i++){
        fprintf(stderr, "\t %d = %g\n", i, test_array[i]);
    }

    return 1;
}