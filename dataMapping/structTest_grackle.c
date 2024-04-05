#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "structTest.h"

#if _OPENMP
    #include <omp.h>
#endif

void print_field(double *field, int field_size, char *field_name){
    fprintf(stdout, "%s = [", field_name);
    for (int i=0; i < field_size; i++){
        fprintf(stdout, "%g,", field[i]);
    }
    fprintf(stdout, "]\n");
}


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

    // Constants
    const double tiny_number = 1e-20;

    // Set size of, and allocate memory for, grackle's fields
    grackle_fields my_fields;
    
    my_fields.grid_rank = 3;
    my_fields.grid_dimension = (int*) malloc(sizeof(int) * my_fields.grid_rank);
    my_fields.grid_start = (int*) malloc(sizeof(int) * my_fields.grid_rank);
    my_fields.grid_end = (int*) malloc(sizeof(int*) * my_fields.grid_rank);
    
    const int dims[3] = {2,2,1};
    const int starts[3] = {0,0,0};
    const int ends[3] = {2,2,1};
    my_fields.field_size = 1;
    for (int i=0; i<my_fields.grid_rank; i++){
        my_fields.grid_dimension[i] = dims[i];
        my_fields.grid_start[i] = starts[i];
        my_fields.grid_end[i] = ends[i];
        my_fields.field_size *= my_fields.grid_dimension[i];
    }

    my_fields.density = (double*) malloc(sizeof(double) * my_fields.field_size);
    my_fields.internal_energy = (double*) malloc(sizeof(double) * my_fields.field_size);
    my_fields.pressure = (double*) malloc(sizeof(double) * my_fields.field_size);

    
    // Allocate values to grackle structs
    grackle_chemistry my_chemistry;
    my_chemistry.gamma = 3/2;
    for (int i=0; i<my_fields.field_size; i++){
        my_fields.pressure[i] = 0.;
        my_fields.density[i] = 0.;
        my_fields.internal_energy[i] = 0.;
    }

    print_field(my_fields.density, my_fields.field_size, "initial density ");
    print_field(my_fields.pressure, my_fields.field_size, "initial pressure ");
    print_field(my_fields.internal_energy, my_fields.field_size, "initial internal energy ");

    //Declare mapper for grackle_fields struct
    #pragma omp declare mapper(grackle_fields_mapper: my_fields f) \
            map(f, f.density[0:f.field_size],\
                f.internal_energy[0:f.field_size],\
                f.pressure[0:f.field_size])

    //Test passing struct to target region
    #pragma omp target enter data map(to:mapper(grackle_fields_mapper))
    #pragma omp target enter data map(to:my_fields.pressure[:my_fields.field_size],\
                                         my_fields.internal_energy[:my_fields.field_size],\
                                         my_fields.density[:my_fields.field_size])

    #pragma omp target
    {
        for (int i=0; i < my_fields.field_size; i++){
            my_fields.pressure[i] = 1.;
            my_fields.density[i] = 1.;
            my_fields.internal_energy[i] = 1.;
        }
    }

    // #pragma omp target exit data map(from:mapper(grackle_fields_mapper))
    #pragma omp target exit data map(from:my_fields.pressure[:my_fields.field_size],\
                                          my_fields.internal_energy[:my_fields.field_size],\
                                          my_fields.density[:my_fields.field_size])

    print_field(my_fields.density, my_fields.field_size, "final density ");
    print_field(my_fields.pressure, my_fields.field_size, "final pressure ");
    print_field(my_fields.internal_energy, my_fields.field_size, " final internal energy ");

    //! BELOW CODE IS HOW THE LOOPS ARE IMPLEMENTED IN GRACKLE CURRENTLY
    
    // Create index helper object
    // const grackle_index_helper index_helper = _build_index_helper(gridRank,gridDimensions,
                                                                // gridStart,gridEnd);

    //TODO  Compile a version of the _inner_range function which can run on the GPU
    // #pragma omp declare target
    // grackle_index_range _inner_range(int outer_index, const grackle_index_helper* ind_helper);
    // #pragma omp end declare target

    // Loop over the outer index
    //TODO  Map the grackle_index_helper struct and the _inner_range function to the GPU.
    //TODO   Then parallelise over the outer loop.

    // for (outer_index = 0; outer_index < index_helper.outer_ind_size; outer_index++){
    // for (int outer_index = 0; outer_index < 5; outer_index++){

        // Compute range of inner loop
        // const grackle_index_range range = _inner_range(outer_index, &index_helper);

        // Loop over inner index
        // for (index = range.start; index <= range.end; index++){

            // Calculate the pressure for each index
            // pressure[index] = ((gamma - 1.0) * density[index] * internal_energy[index]);
            // if (pressure[index] < tiny_number) pressure[index] = tiny_number;
        // }  
    // }

    return 1;
}