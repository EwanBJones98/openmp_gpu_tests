#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#if _OPENMP
    #include <omp.h>
#endif

// Function which checks that the GPU and OpenMP are talking to eachother.
void check_gpu()
{
    int active;
    #pragma omp target map(tofrom:active)
    {
        active = omp_is_initial_device();
    }
    fprintf(stdout, "\n");
    if (!active){
        fprintf(stdout, "-- GPU is active --\n\n");
    } else {
        fprintf(stdout, "-- GPU is *inactive* --\nTerminating...\n\n");
        exit(0);
    }
}

// Simple structure which represents an N-dimensional vector.
typedef struct
{
    int size;
    double *components;
} vector;

// Function which prints a given vector struct.
void print_vector(vector *input_vector)
{
    fprintf(stdout, "[");
    for (int i=0; i<input_vector->size; i++){
        fprintf(stdout, "%g", input_vector->components[i]);
        if (i != input_vector->size - 1){
            fprintf(stdout, ",");
        }
    }
    fprintf(stdout, "]\n");
}
    
int main(int argc, char *argv[]){

    // Check that openmp can interface with the gpu
    check_gpu();

    // Create a 3-dimenisonal vector [1,1,1]
    vector my_vector;
    my_vector.size = 3;
    my_vector.components = (double*) malloc(sizeof(double) * my_vector.size);
    for (int i=0; i<my_vector.size; i++){
        my_vector.components[i] = 1.;
    }

    // Print initial vector
    fprintf(stdout, "Inital vector:\t");
    print_vector(&my_vector);

    //! The following declares a custom mapper for the struct, but only works in omp>5.0
    // #pragma omp target declare mapper(my_vector v) map(v.size, v.components[:v.size])

    // Map vector to the GPU
    #pragma omp target enter data map(my_vector.size, my_vector.components[:my_vector.size])

    // Modify the vector on the GPU
    #pragma omp target
    {
        for (int i=0; i<my_vector.size; i++){
            my_vector.components[i] *= 2;
        }
    }

    // Map updated vector from the GPU
    #pragma omp target exit data map(my_vector.components[:my_vector.size])

    //Print final vector
    fprintf(stdout, "Final vector:\t");
    print_vector(&my_vector);

    return 1;
}