#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "structTest_grackle.h"

#if _OPENMP
    #include <omp.h>
#endif

void print_field(double *field, int field_size, char *field_name)
{
    fprintf(stdout, "%s = [", field_name);
    for (int i=0; i < field_size; i++){
        fprintf(stdout, "%g,", field[i]);
    }
    fprintf(stdout, "]\n");
}

double* copy_array_from_struct(double *array, int array_size)
{
    double *field_copy;
    field_copy = (double*) malloc(sizeof(double) * array_size);

    for (int i=0; i<array_size; i++){
        field_copy[i] = array[i];
    }

    return field_copy;
}

void copy_array_to_struct(double *array, int array_size, double *struct_array, int struct_array_size)
{
    if (array_size != struct_array_size){
        fprintf(stdout, "Array sizes do not match! Exiting...");
        exit(0);
    }

    for (int i=0; i<array_size; i++){
        struct_array[i] = array[i];
    }
}


void check_gpu()
{
    int active;
    #pragma omp target map(tofrom:active)
    {
        active = omp_is_initial_device();
    }
    if (active == 0){
        fprintf(stdout, "\n GPU is active \n");
    } else {
        fprintf(stdout, "\n GPU not active! Quitting...");
        exit(0);
    }
}

void initialise_chemistry(grackle_chemistry *my_chemistry)
{
    my_chemistry->gamma = 3/2;
}

void initialise_fields(grackle_fields *my_fields)
{
    my_fields->grid_rank = 3;
    my_fields->grid_dimension = (int*) malloc(sizeof(int) * my_fields->grid_rank);
    my_fields->grid_start = (int*) malloc(sizeof(int) * my_fields->grid_rank);
    my_fields->grid_end = (int*) malloc(sizeof(int*) * my_fields->grid_rank);
    
    const int dims[3] = {2,2,1};
    const int starts[3] = {0,0,0};
    const int ends[3] = {2,2,1};
    my_fields->field_size = 1;
    for (int i=0; i<my_fields->grid_rank; i++){
        my_fields->grid_dimension[i] = dims[i];
        my_fields->grid_start[i] = starts[i];
        my_fields->grid_end[i] = ends[i];
        my_fields->field_size *= my_fields->grid_dimension[i];
    }

    my_fields->density = (double*) malloc(sizeof(double) * my_fields->field_size);
    my_fields->internal_energy = (double*) malloc(sizeof(double) * my_fields->field_size);
    my_fields->pressure = (double*) malloc(sizeof(double) * my_fields->field_size);

    for (int i=0; i<my_fields->field_size; i++){
        my_fields->pressure[i] = 0.;
        my_fields->density[i] = 1.e-10;
        my_fields->internal_energy[i] = 1.e10;
    }
}


int main(int arvc, char *argv[])
{

    const double tiny_number = 1e-20;

    // Check GPU is activated correctly
    check_gpu();

    // Initialise fields to their default values
    grackle_fields my_fields;
    initialise_fields(&my_fields);
 
    // Initialise chemistry parameters to their defualt values
    grackle_chemistry my_chemistry;
    initialise_chemistry(&my_chemistry);
    
    // Print initial field values
    print_field(my_fields.density, my_fields.field_size, "initial density ");
    print_field(my_fields.pressure, my_fields.field_size, "initial pressure ");
    print_field(my_fields.internal_energy, my_fields.field_size, "initial internal energy ");

    // As I cannot yet declare a custom mapper until openmp >5.0 is installed, I will have to
    //  copy the neccessary fields from the structure into their own arrays, which is very inefficient.
    double *my_fields_density, *my_fields_pressure, *my_fields_internal_energy;
    int my_field_size;
    my_field_size = my_fields.field_size;
    my_fields_density = copy_array_from_struct(my_fields.density, my_field_size);
    my_fields_pressure = copy_array_from_struct(my_fields.pressure, my_field_size);
    my_fields_internal_energy = copy_array_from_struct(my_fields.internal_energy, my_field_size);

    // Map the arrays to the device
    #pragma omp target enter data map(to:my_field_size, my_fields_density[:my_field_size],\
                                         my_fields_pressure[:my_field_size],\
                                         my_fields_internal_energy[:my_field_size])

    // Change the values of the arrays on the device
    #pragma omp target teams distribute parallel for
    for(int i=0; i<my_field_size; i++){
        my_fields_density[i] = 4.;
        my_fields_pressure[i] = 4.;
        my_fields_internal_energy[i] = 4.;
    }

    // Map the modified arrays from the device
    #pragma omp target exit data map(from:my_fields_density[:my_field_size],\
                                          my_fields_pressure[:my_field_size],\
                                          my_fields_internal_energy[:my_field_size])

    // Copy the arrays back into the grackle_fields structure
    copy_array_to_struct(my_fields_density, my_field_size, my_fields.density, my_fields.field_size);
    copy_array_to_struct(my_fields_pressure, my_field_size, my_fields.pressure, my_fields.field_size);
    copy_array_to_struct(my_fields_internal_energy, my_field_size, my_fields.internal_energy, my_fields.field_size);

    // Print final field values
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