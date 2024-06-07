#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "grackle.h"
#include "grackle_types.h"
#include "grackle_chemistry_data.h"
#include "phys_constants.h"
#include "grackle_macros.h"
#include "index_helper.h"

#include "calculate_pressure_gpu.h"

#ifdef _OPENMP
    #include <omp.h>
#else
    #include <time.h>
    static int omp_get_thread_num() {return 0;}
    static int omp_get_num_threads() {return 1;}
    static double omp_get_wtime() {return (clock() / CLOCKS_PER_SEC);}
#endif

// Function prototype for writing out results to file
int write_timings(int num_iterations, double mean_time, double stdev, int num_teams,
                    int num_threads, char mode[10], grackle_field_data *my_fields,
                    chemistry_data *my_chemistry, int new_file, char filename[100]);

double tiny_number = 1e-20;

int calculate_pressure_gpu(chemistry_data *my_chemistry,
                           chemistry_data_storage *my_rates,
                           code_units *my_units,
                           grackle_field_data *my_fields,
                           gr_float *pressure,
                           double pressure_units,
                           int num_iter, double *timings)
{
    if (!my_chemistry->use_grackle) return 1;
    
    double temperature_units, GammaInverse;
    if (my_chemistry->primordial_chemistry > 1)
    {
        temperature_units = get_temperature_units(my_units);
        GammaInverse      = 1.0 / (my_chemistry->Gamma - 1.0);
    }

    const grackle_index_helper ind_helper = _build_index_helper(my_fields);

    int field_length = 1;
    for (int i=0; i < my_fields->grid_rank; i++)
    {
        field_length *= my_fields->grid_dimension[i];
    }

    // Map data to the GPU
    #if defined(_OPENMP) && defined(GPU)   
        enter_calculate_pressure(pressure, temperature_units, GammaInverse,
                                tiny_number, my_chemistry, my_fields,
                                my_units, &ind_helper);
    #endif

    // Precompute indicies of buffer zones inside the 1D array
    #ifdef USE_BUFFER_ZONES
    #pragma omp begin declare target
        int buffer_zones[field_length], index;
    #pragma omp end declare target
    #pragma omp target teams distribute parallel for collapse(3)
    for (int i=0; i<my_fields->grid_dimensions[0]; i++){
        for (int j=0; j<my_fields->grid_dimensions[1]; j++){
            for (int k=0; k<my_fields->grid_dimensions[2]; k++){
                index = i + (my_fields->grid_dimensions[1] *
                                (j + my_fields->grid_dimensions[2] * k));
                if (i < my_fields->grid_start[0] || i > my_fields->grid_end[0]){
                    buffer_zones[index] = 1;
                } else if (j < my_fields->grid_start[1] || j > my_fields->grid_end[1]){
                    buffer_zones[index] = 1;
                } else if (k < my_fields->grid_start[2] || k > my_fields->grid_end[2]){
                    buffer_zones[index] = 1;
                } else {
                    buffer_zones[index] = 0;
                }    
            }
        }
    }
    #endif

    // Begin timing iterations
    double start_time;
    for (int iter=0; iter<num_iter; iter++)
    {

        start_time = omp_get_wtime();

        #if defined(_OPENMP) && defined(CPU)
            #pragma omp parallel for schedule(runtime)
        #elif defined(_OPENMP) && defined(GPU)
            #pragma omp target teams distribute parallel for
        #endif
        for (int index=0; index<field_length; index++)
        {   
            #ifdef USE_BUFFER_ZONES
                if (buffer_zones[index] == 1) continue;
            #endif

            pressure[index] = (my_chemistry->Gamma -1.0) * my_fields->density[index]
                                * my_fields->internal_energy[index];

            if (my_chemistry->primordial_chemistry > 1)
            {
                double number_density = 0.25 * (my_fields->HeI_density[index]
                                                + my_fields->HeII_density[index]
                                                + my_fields->HeIII_density[index])
                                        + my_fields->HI_density[index]
                                        + my_fields->HII_density[index]
                                        + my_fields->HM_density[index]
                                        + my_fields->e_density[index];

                double nH2 = 0.5 * (my_fields->H2I_density[index] + my_fields->H2II_density[index]);

                // First, approximate temperature
                if (number_density == 0) number_density = tiny_number;
                double temp = max(temperature_units * pressure[index] / (number_density + nH2), 1);

                // Only do full computation if there is a reasonable amount of H2.
                // The second term in GammaH2Inverse accounts for the vibrational
                //  degrees of freedom.

                double GammaH2Inverse = 0.5 * 5.0;
                if (nH2 / number_density > 1e-3)
                {
                    double x = 6100.0 / temp;
                    if (x < 10.0)
                    {
                        GammaH2Inverse = 0.5 * (5 + 2.0 * POW(x, 2)
                                                * exp(x)/POW(exp(x) - 1.0, 2));
                    }
                }

                double Gamma1 = 1.0 + (nH2 + number_density)
                        / (nH2 * GammaH2Inverse + number_density * GammaInverse);

                // Correct pressure with improved Gamma.
                pressure[index] *= (Gamma1 - 1.0) / (my_chemistry->Gamma - 1.0);
            } // End primordial_chemistry > 1

            pressure[index] = max(tiny_number, pressure[index]);
        } // End loop over outer index

        timings[iter] = omp_get_wtime() - start_time;

        #if defined(_OPENMP) && defined(GPU)
            #pragma omp target update from(pressure[:field_length])
        #endif

        // Check that the pressure has been updated and reset for next loop
        for (int i=0; i<field_length; i++)
        {
            if (pressure[i] == -1) return 0;

            // Print result of pressure calculation to terminal every so often
            if (iter % 100 == 0 && i == 0)
            {
                fprintf(stdout, "Iteration %d: pressure = %g\n", iter, pressure[0] * pressure_units);
            }

            pressure[i] = -1;
        }

        #if defined(_OPENMP) && defined(GPU)
            #pragma omp target update to(pressure[:field_length])
        #endif

    } // End timing iterations

    #if defined(_OPENMP) && defined(GPU)
        exit_calculate_pressure(pressure, temperature_units, GammaInverse,
                                tiny_number, my_chemistry, my_fields,
                                my_units, &ind_helper);
    #endif
   
    return 1;
}

void initialise_grackle_structs(int grid_rank, int grid_dimensions[],
                                int primordial_chemistry,
                                double initial_redshift,
                                double temperature,
                                chemistry_data *my_chemistry,
                                chemistry_data_storage *my_rates,
                                code_units *my_units,
                                grackle_field_data *my_fields)
{
    // >>> Set up unit system <<<
    my_units->comoving_coordinates = 0; // 1 if cosmological sim, 0 if not
    my_units->density_units = 1.67e-24;
    my_units->length_units = 1.0;
    my_units->time_units = 1.0e12;
    my_units->a_units = 1.0; // units for the expansion factor
    my_units->a_value = 1. / (1. + initial_redshift) / my_units->a_units;
    set_velocity_units(my_units);
    
    // >>> Initialise chemistry parameters and rates <<<
    local_initialize_chemistry_data(my_chemistry, my_rates, my_units);
    set_default_chemistry_parameters(my_chemistry);

    my_chemistry->use_grackle = 1;
    my_chemistry->primordial_chemistry = primordial_chemistry;

    // >>> Initialise fields <<<
    my_fields->grid_rank = 3;
    my_fields->grid_dimension = malloc(my_fields->grid_rank * sizeof(int));
    my_fields->grid_start     = malloc(my_fields->grid_rank * sizeof(int));
    my_fields->grid_end       = malloc(my_fields->grid_rank * sizeof(int));

    int field_length = 1;
    for (int i=0; i<my_fields->grid_rank; i++)
    {
        my_fields->grid_dimension[i] = grid_dimensions[i];
        my_fields->grid_start[i]     = 0;
        my_fields->grid_end[i]       = my_fields->grid_dimension[i] - 1;

        field_length *= grid_dimensions[i];
    }
    my_fields->density         = malloc(field_length * sizeof(double));
    my_fields->internal_energy = malloc(field_length * sizeof(double));
    my_fields->HI_density      = malloc(field_length * sizeof(double));
    my_fields->HII_density     = malloc(field_length * sizeof(double));
    my_fields->HM_density      = malloc(field_length * sizeof(double));
    my_fields->HeI_density     = malloc(field_length * sizeof(double));
    my_fields->HeII_density    = malloc(field_length * sizeof(double));
    my_fields->HeIII_density   = malloc(field_length * sizeof(double));
    my_fields->H2I_density     = malloc(field_length * sizeof(double));
    my_fields->H2II_density    = malloc(field_length * sizeof(double));
    my_fields->e_density       = malloc(field_length * sizeof(double));

    for (int i=0; i<field_length; i++)
    {
        my_fields->density[i]         = 1.0;
        my_fields->internal_energy[i] = temperature / get_temperature_units(my_units);
        my_fields->HI_density[i]      = my_chemistry->HydrogenFractionByMass * my_fields->density[i];
        my_fields->HII_density[i]     = tiny_number * my_fields->density[i];
        my_fields->HM_density[i]      = tiny_number * my_fields->density[i];
        my_fields->HeI_density[i]     = (1.0 - my_chemistry->HydrogenFractionByMass) * my_fields->density[i];
        my_fields->HeII_density[i]    = tiny_number * my_fields->density[i];
        my_fields->HeIII_density[i]   = tiny_number * my_fields->density[i];
        my_fields->H2I_density[i]     = tiny_number * my_fields->density[i];
        my_fields->H2II_density[i]    = tiny_number * my_fields->density[i];
        my_fields->e_density[i]       = tiny_number * my_fields->density[i];


        //! These are only here to modify the values for testing. Values are not physical.
        double h_fraction = 0.5;
        my_fields->HI_density[i]      = h_fraction * my_fields->density[i];
        my_fields->HII_density[i]     = 0.05 * my_fields->density[i];
        my_fields->HM_density[i]      = 0.01 * my_fields->density[i];
        my_fields->HeI_density[i]     = (1.0 - h_fraction - 0.2) * my_fields->density[i];
        my_fields->HeII_density[i]    = 0.01 * my_fields->density[i];
        my_fields->HeIII_density[i]   = 0.01 * my_fields->density[i];
        my_fields->H2I_density[i]     = 0.1 * my_fields->density[i];
        my_fields->H2II_density[i]    = 0.01 * my_fields->density[i];
        my_fields->e_density[i]       = 0.01 * my_fields->density[i];
    }
}

void free_grackle_fields(grackle_field_data *my_fields)
{
    free(my_fields->density);
    free(my_fields->internal_energy);
    free(my_fields->HI_density);
    free(my_fields->HII_density);
    free(my_fields->HM_density);
    free(my_fields->HeI_density);
    free(my_fields->HeII_density);
    free(my_fields->HeIII_density);
    free(my_fields->H2I_density);
    free(my_fields->H2II_density);
    free(my_fields->e_density);
}


int main(int argc, char *argv[])
{

    chemistry_data my_chemistry;
    chemistry_data_storage my_rates;
    code_units my_units;
    grackle_field_data my_fields;

    //! >>> USER SETTINGS <<<
    double initial_redshift = 2;
    double temperature = 1000;
    int primordial_chemistry = 2;

    int grid_rank = 3;
    int grid_dimensions[]   = {100,100,100};

    int num_timing_iter = 10000;
    

    int overwrite=0;
    char filepath[] = "/home/ejones/codes/openmp_gpu_tests/grackle_benchmarks/timings.txt";
    //! >>> ---------- <<<

    // >>> Get parallelism info <<<
    int num_threads, num_teams;
    char mode[10];
    #ifdef _OPENMP
        #ifdef CPU
            strcpy(mode, "CPU");
            num_teams = 1;
            omp_cpu_info(&num_threads);
            fprintf(stdout, "Running with CPU parallelism!\n\tnumber of threads: %d\n\n", num_threads);
        #elif GPU
            int gpu_active;
            strcpy(mode, "GPU");
            omp_gpu_info(&gpu_active, &num_threads, &num_teams);
            if (gpu_active == 0)
            {
                fprintf(stdout, "Running with GPU parallelism!\n\tnumber of threads: %d\n\tnumber of teams:%d\n\n",
                            num_threads, num_teams);
            } else {
                strcpy(mode, "serial");
                fprintf(stdout, "GPU inactive! Exiting...\n");
            }
        #endif
    #else
        num_threads = 1;
        num_teams   = 1;
        fprintf(stdout, "Running in serial on the CPU!\n\n");
    #endif
    // >>> ---------- <<<

    initialise_grackle_structs(grid_rank, grid_dimensions, primordial_chemistry, initial_redshift, temperature,
                               &my_chemistry, &my_rates, &my_units, &my_fields);
    
    int field_length = 1;
    for (int i=0; i<grid_rank; i++)
        field_length *= grid_dimensions[i];

    double *pressure, pressure_units;
    pressure = malloc(field_length * sizeof(double));
    pressure_units = my_units.density_units * pow(my_units.velocity_units, 2);

    for (int i=0; i<field_length; i++)
        pressure[i] = -1;

    double timings[num_timing_iter];
    calculate_pressure_gpu(&my_chemistry, &my_rates, &my_units, &my_fields,
                           pressure, pressure_units, num_timing_iter, timings);

    double avg_time = 0, stdev=0;
    for (int i=0; i<num_timing_iter; i++)
        avg_time += timings[i] / num_timing_iter;
    for (int i=0; i<num_timing_iter; i++)
        stdev += pow(timings[i] - avg_time, 2);
    stdev = sqrt(stdev/num_timing_iter);


    write_timings(num_timing_iter, avg_time, stdev, num_teams, num_threads,
                    mode, &my_fields, &my_chemistry, overwrite, filepath);

    local_free_chemistry_data(&my_chemistry, &my_rates);
    free_grackle_fields(&my_fields);
        
    return 1;
}