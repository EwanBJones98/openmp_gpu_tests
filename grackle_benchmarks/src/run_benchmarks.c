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

#include "write_timings.h"
#include "worksharing_info.h"

#include "calculate_pressure.h"
#include "calculate_gamma.h"
#include "calculate_temperature.h"

#ifdef _OPENMP
    #include <omp.h>
#else
    #include <time.h>
    static int omp_get_thread_num() {return 0;}
    static int omp_get_num_threads() {return 1;}
    static double omp_get_wtime() {return (clock() / CLOCKS_PER_SEC);}
#endif



int read_command_line_arguments(int argc, char *argv[], int *primordial_chemistry, int *grid_dimension_i,
                                 int *numThreadsPerTeam, char *timings_dir)
{
    if (argc != 5)
    {
        fprintf(stderr, "** WRONG NUMBER OF COMMAND LINE ARGUMENTS **\n");
        fprintf(stderr, "Expected 4 -- Received %d\n", argc-1);
        exit(0);
    }

    *primordial_chemistry = (int) atol(argv[1]);
    *grid_dimension_i     = (int) atol(argv[2]);
    *numThreadsPerTeam    = (int) atol(argv[3]);
    strcpy(timings_dir, argv[4]);

    // If the number of threads per team == -1 then allow to compiler to use its default number
    if (*numThreadsPerTeam == -1)
    {
        int buffer1, buffer2;
        gpu_default_worksharing_info(&buffer1, &buffer2, numThreadsPerTeam);
    }

    return 1;
}


int initialise_grackle_structs(int primordial_chemistry, double initial_density,
                                double initial_temperature, double initial_redshift,
                                int grid_rank, int *grid_dimensions,
                                chemistry_data *my_chemistry, chemistry_data_storage *my_rates,
                                grackle_field_data *my_fields, code_units *my_units)
{
    double tiny_number = 1e-20;

    // >>> Set up unit system <<<
    my_units->comoving_coordinates = 0; // 1 if cosmological sim, 0 if not
    my_units->density_units = 1.67e-24;
    my_units->length_units = 1.0;
    my_units->time_units = 1.0e12;
    my_units->a_units = 1.0; // units for the expansion factor
    my_units->a_value = 1. / (1. + initial_redshift) / my_units->a_units;
    set_velocity_units(my_units);

    double temperature_units = get_temperature_units(my_units);
    
    // >>> Initialise chemistry parameters and rates <<<
    local_initialize_chemistry_data(my_chemistry, my_rates, my_units);
    set_default_chemistry_parameters(my_chemistry);

    my_chemistry->use_grackle = 1;
    my_chemistry->primordial_chemistry = primordial_chemistry;

    // >>> Initialise fields <<<
    my_fields->grid_rank      = grid_rank;
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
        my_fields->density[i]         = initial_density;
        my_fields->internal_energy[i] = initial_temperature / temperature_units;
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
        my_fields->HI_density[i]    = h_fraction * my_fields->density[i];
        my_fields->HII_density[i]   = 0.05 * my_fields->density[i];
        my_fields->HM_density[i]    = 0.01 * my_fields->density[i];
        my_fields->HeI_density[i]   = (1.0 - h_fraction - 0.2) * my_fields->density[i];
        my_fields->HeII_density[i]  = 0.01 * my_fields->density[i];
        my_fields->HeIII_density[i] = 0.01 * my_fields->density[i];
        my_fields->H2I_density[i]   = 0.1 * my_fields->density[i];
        my_fields->H2II_density[i]  = 0.01 * my_fields->density[i];
        my_fields->e_density[i]     = 0.01 * my_fields->density[i];
    }
    return 1;
}

int free_grackle_structs(chemistry_data *my_chemistry, chemistry_data_storage *my_rates,
                          grackle_field_data *my_fields, code_units *my_units)
{
    local_free_chemistry_data(my_chemistry, my_rates);

    free(my_units);

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


int run_benchmark(void (*calc_func)(double *field,
                                    int field_length,
                                    chemistry_data *my_chemistry,
                                    chemistry_data_storage *my_rates,
                                    grackle_field_data *my_fields,
                                    code_units *my_units,
                                    int nTeams,
                                    int nThreadsPerTeam),
                  void (*enter_gpu)(double *field,
                                     chemistry_data *my_chemistry,
                                     grackle_field_data *my_fields,
                                     code_units *my_units),
                  void (*exit_gpu)(double *field,
                                    chemistry_data *my_chemistry,
                                    grackle_field_data *my_rates,
                                    code_units *my_units),
                  chemistry_data *my_chemistry,
                  chemistry_data_storage *my_rates,
                  grackle_field_data *my_fields,
                  code_units *my_units,
                  int num_timing_iter,
                  double *calc_times,
                  double *data_times,
                  int nTeams,
                  int nThreadsPerTeam)
{
    if (!my_chemistry->use_grackle) return 1;
    
    int field_length = 1;
    for (int i=0; i < my_fields->grid_rank; i++)
    {
        field_length *= my_fields->grid_dimension[i];
    }

    double *my_values;
    my_values = malloc(field_length * sizeof(double));

    double calc_start_time, calc_elapsed_time;
    double total_start_time, total_elapsed_time;

    for (int iter=0; iter<num_timing_iter; iter++)
    {
        
        total_start_time = omp_get_wtime();

        #if defined(GPU) && defined(_OPENMP)
            enter_gpu(my_values, my_chemistry, my_fields, my_units);
        #endif

        calc_start_time = omp_get_wtime();
        calc_func(my_values, field_length, my_chemistry, my_rates,
                    my_fields, my_units, nTeams, nThreadsPerTeam);
        calc_elapsed_time = omp_get_wtime() - calc_start_time;

        #if defined(GPU) && defined(_OPENMP)
            exit_gpu(my_values, my_chemistry, my_fields, my_units);
        #endif

        total_elapsed_time = omp_get_wtime() - total_start_time;

        calc_times[iter] = calc_elapsed_time;
        data_times[iter] = total_elapsed_time - calc_elapsed_time;
    }

    return 1;
}


int main(int argc, char *argv[])
{
    code_units *my_units;
    chemistry_data *my_chemistry;
    grackle_field_data *my_fields;
    chemistry_data_storage *my_rates;

    // >>> Read command line args <<<
    int primordial_chemistry, grid_dim_i, numThreadsPerTeam;
    char timing_dir[200];

    read_command_line_arguments(argc, argv, &primordial_chemistry, &grid_dim_i,
                                 &numThreadsPerTeam, timing_dir);

    // Read environment variables also
    char parallel_mode[10];
    get_parallel_mode(parallel_mode);
    // >>> ---------------------- <<<

    // >>> Set worksharing parameters <<<
    int max_threads_per_team = 1024;
    int max_threads_per_sm   = 1536;
    int num_sm               = 142;
    int max_threads = max_threads_per_sm * num_sm;
    
    int numTeams = (int) max_threads / numThreadsPerTeam;

    // Check that the GPU is active and can allocate the desired number of threads/teams
    if (omp_get_default_device() != 0)
    {
        fprintf(stderr, "GPU inactive!\nExiting...\n");
        exit(0);
    }
    if (!check_gpu_worksharing(&numTeams, &numThreadsPerTeam))
    {
        fprintf(stderr, "Error allocating requested number of teams/threads per team");
        fprintf(stderr, "please see stdout stream for more information.\nExiting...\n");
        exit(0);
    }
    // >>> ------------------------- <<<

    // >>> Set grackle parameters <<<
    double redshift    = 2;
    double density     = 1e-24;
    double temperature = 1e5;

    int grid_rank = 3;
    int grid_dimensions[] = {grid_dim_i,100,100};
    // >>> ---------------------- <<<

    initialise_grackle_structs(primordial_chemistry, density, temperature, redshift, grid_rank,
                                grid_dimensions, my_chemistry, my_rates, my_fields, my_units);

    // >>> Setup timing framework <<<
    char savepath_buffer[300]; // Used for timing output path
    int num_timing_iter = 1000;
    double *calc_times, *data_times;

    calc_times = malloc(num_timing_iter * sizeof(double));
    data_times = malloc(num_timing_iter * sizeof(double));
    // >>> ---------------------- <<<
    

    // >>> Run pressure test <<<
    run_benchmark(calculate_pressure, enter_calculate_pressure, exit_calculate_pressure,
                   my_chemistry, my_rates, my_fields, my_units, num_timing_iter, calc_times,
                   data_times, numTeams, numThreadsPerTeam);

    strcpy(savepath_buffer, timing_dir);
    strcat(savepath_buffer, "pressure.txt");

    write_timings(num_timing_iter, calc_times, data_times, numTeams, numThreadsPerTeam,
                   parallel_mode, my_fields, my_chemistry, 0, savepath_buffer);
    // >>> ----------------- <<<


    // >>> Run gamma test <<<
    run_benchmark(calculate_gamma, enter_calculate_gamma, exit_calculate_gamma,
                   my_chemistry, my_rates, my_fields, my_units, num_timing_iter, calc_times,
                   data_times, numTeams, numThreadsPerTeam);

    strcpy(savepath_buffer, timing_dir);
    strcat(savepath_buffer, "gamma.txt");

    write_timings(num_timing_iter, calc_times, data_times, numTeams, numThreadsPerTeam,
                   parallel_mode, my_fields, my_chemistry, 0, savepath_buffer);
    // >>> ----------------- <<<


    // >>> Run temperature test <<<
    run_benchmark(calculate_temperature, enter_calculate_temperature, exit_calculate_temperature,
                   my_chemistry, my_rates, my_fields, my_units, num_timing_iter, calc_times,
                   data_times, numTeams, numThreadsPerTeam);

    strcpy(savepath_buffer, timing_dir);
    strcat(savepath_buffer, "temperature.txt");

    write_timings(num_timing_iter, calc_times, data_times, numTeams, numThreadsPerTeam,
                   parallel_mode, my_fields, my_chemistry, 0, savepath_buffer);
    // >>> ----------------- <<<

 
    // >>> Cleanup memory <<<
    free_grackle_structs(my_chemistry, my_rates, my_fields, my_units);
    free(calc_times);
    free(data_times);
    // >>> -------------- <<<

    return 0;
}