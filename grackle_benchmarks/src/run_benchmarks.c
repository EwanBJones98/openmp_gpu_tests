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

#include "utility.h"

#include "worksharing_info.h"

#include "write_timings.h"

#include "calculate_pressure.h"
#include "calculate_gamma.h"
#include "calculate_temperature.h"
#include "gpu_data_handling.h"

#ifdef _OPENMP
    #include <omp.h>
#else
    #include <time.h>
    static int omp_get_thread_num() {return 0;}
    static int omp_get_num_threads() {return 1;}
    static double omp_get_wtime()
        {
            struct timespec t;
            timespec_get(&t, TIME_UTC);
            return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
        }
#endif


int read_command_line_arguments(int argc, char *argv[], int *primordial_chemistry, int *grid_dimension_i,
                                 int *numTeams, char *timings_dir)
{
    #if defined(GPU) || defined(CPU)
        if (argc != 5)
        {
            fprintf(stderr, "** WRONG NUMBER OF COMMAND LINE ARGUMENTS **\n");
            fprintf(stderr, "Expected 4 -- Received %d\n", argc-1);
            exit(0);
        }
    #else
        if (argc != 4)
        {
           fprintf(stderr, "** WRONG NUMBER OF COMMAND LINE ARGUMENTS **\n");
           fprintf(stderr, "Expected 3 -- Received %d\n", argc-1);
           exit(0); 
        }
    #endif

    *primordial_chemistry = (int) atol(argv[1]);
    *grid_dimension_i     = (int) atol(argv[2]);

    #if defined(GPU) || defined(CPU)
        *numTeams             = (int) atol(argv[3]);
        strcpy(timings_dir, argv[4]);
    #else
        *numTeams = 1;
        strcpy(timings_dir, argv[3]);
    #endif

    #if defined(GPU)
        // If the number of threads per team == -1 then allow to compiler to use its default number
        if (*numTeams == -1)
        {
            int buffer1, buffer2;
            gpu_default_worksharing_info(&buffer1, &numTeams, &buffer2);
        }
    #endif

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
    my_units->density_units        = 1.67e-24;
    my_units->length_units         = 1.0;
    my_units->time_units           = 1.0e12;
    my_units->a_units              = 1.0; // units for the expansion factor
    my_units->a_value              = 1. / (1. + initial_redshift) / my_units->a_units;
    set_velocity_units(my_units);
    set_temperature_units(my_units);
    
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
        my_fields->internal_energy[i] = initial_temperature / my_units->temperature_units;
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

int run_benchmark(chemistry_data *my_chemistry,
                  chemistry_data_storage *my_rates,
                  grackle_field_data *my_fields,
                  code_units *my_units,
                  int nTeams,
                  int nThreadsPerTeam,
                  int num_timing_iter,
                  char *timing_dir)
{
    if (!my_chemistry->use_grackle) return 1;
    
    int field_length = 1;
    for (int i=0; i < my_fields->grid_rank; i++)
    {
        field_length *= my_fields->grid_dimension[i];
    }

    double *my_field_buffer;
    my_field_buffer = malloc(field_length * sizeof(double));

    double calc_start_time, calc_end_time;
    double pressure_calc_times[num_timing_iter];
    double temperature_calc_times[num_timing_iter];
    double gamma_calc_times[num_timing_iter];

    #if defined(GPU) && defined(_OPENMP)
        double data_start_time, data_end_time;
        double data_mapping_times[num_timing_iter];
    #endif

    for (int iter=0; iter<num_timing_iter; iter++)
    {
        
        #if defined(GPU) && defined(_OPENMP)
            // fprintf(stdout, "Entering target region...\n");
            data_start_time = omp_get_wtime();
            enter_gpu(my_field_buffer, my_chemistry, my_fields, my_units);
            data_end_time            = omp_get_wtime();
            data_mapping_times[iter] = data_end_time - data_start_time;
        #endif

        // Run benchmark: calculate_pressure
        calc_start_time = omp_get_wtime();
        calculate_pressure(my_field_buffer, field_length, my_chemistry, my_rates,
                            my_fields, my_units, nTeams, nThreadsPerTeam);
        calc_end_time             = omp_get_wtime();
        pressure_calc_times[iter] = calc_end_time - calc_start_time;

        // Run benchmark: calculate_temperature
        calc_start_time = omp_get_wtime();
        calculate_temperature(my_field_buffer, field_length, my_chemistry, my_rates,
                                my_fields, my_units, nTeams, nThreadsPerTeam);
        calc_end_time                = omp_get_wtime();
        temperature_calc_times[iter] = calc_end_time - calc_start_time;

        // Run benchmark: calculate_gamma
        calc_start_time = omp_get_wtime();
        calculate_gamma(my_field_buffer, field_length, my_chemistry, my_rates,
                                my_fields, my_units, nTeams, nThreadsPerTeam);
        calc_end_time          = omp_get_wtime();
        gamma_calc_times[iter] = calc_end_time - calc_start_time;

        #if defined(GPU) && defined(_OPENMP)
            // fprintf(stdout, "Exiting target region...\n");
            data_start_time = omp_get_wtime();
            exit_gpu(my_field_buffer, my_chemistry, my_fields, my_units);
            data_end_time             = omp_get_wtime();
            data_mapping_times[iter] += data_end_time - data_start_time;
        #endif
    }

    // Write benchmarks to appropriate files
    char savepath_buffer[300], parallel_mode[10];
    get_parallel_mode(parallel_mode);

    strcpy(savepath_buffer, timing_dir);
    strcat(savepath_buffer, "pressure.txt");
    write_timings(num_timing_iter, pressure_calc_times, nTeams, nThreadsPerTeam,
                    parallel_mode, my_fields, my_chemistry, 0, savepath_buffer);

    strcpy(savepath_buffer, timing_dir);
    strcat(savepath_buffer, "temperature.txt");
    write_timings(num_timing_iter, temperature_calc_times, nTeams, nThreadsPerTeam,
                    parallel_mode, my_fields, my_chemistry, 0, savepath_buffer);

    strcpy(savepath_buffer, timing_dir);
    strcat(savepath_buffer, "gamma.txt");
    write_timings(num_timing_iter, gamma_calc_times, nTeams, nThreadsPerTeam,
                    parallel_mode, my_fields, my_chemistry, 0, savepath_buffer);

    #if defined(GPU) && defined(_OPENMP)
        strcpy(savepath_buffer, timing_dir);
        strcat(savepath_buffer, "data_mapping.txt");
        write_timings(num_timing_iter, data_mapping_times, nTeams, nThreadsPerTeam,
                        parallel_mode, my_fields, my_chemistry, 0, savepath_buffer);
    #endif

    free(my_field_buffer);

    return 1;
}


int main(int argc, char *argv[])
{
    code_units my_units;
    chemistry_data my_chemistry;
    grackle_field_data my_fields;
    chemistry_data_storage my_rates;

    // >>> Read command line args <<<
    int primordial_chemistry, grid_dim_i, numTeams;
    char timing_dir[200];

    read_command_line_arguments(argc, argv, &primordial_chemistry, &grid_dim_i,
                                 &numTeams, timing_dir);

    // >>> Set worksharing parameters <<<
    int max_threads_per_team = 1024;
    int max_threads_per_sm   = 1536;
    int num_sm               = 142;
    int max_threads          = max_threads_per_sm * num_sm;

    int nThreadsPerTeam = (int) max_threads / numTeams;

    // Check that the GPU is active and can allocate the desired number of threads/teams
    #if defined(GPU)
        if (omp_get_default_device() != 0)
        {
            fprintf(stderr, "GPU inactive!\nExiting...\n");
            exit(0);
        }

        if (!check_gpu_worksharing(&numTeams, &nThreadsPerTeam))
        {
            fprintf(stderr, "Error allocating requested number of teams/threads!\n");
            fprintf(stderr, "See stdout stream for more information. Exiting...\n");
            exit(0);
        }
    // Check the CPU worksharing is running on the correct number of cores
    #elif defined(CPU)
        nThreadsPerTeam = 1;
        if (!check_cpu_worksharing(&numTeams)) exit(0);
    // Set the number of teams and threads for serial run
    #else
        numTeams        = 1;
        nThreadsPerTeam = 1;
    #endif
    // >>> ------------------------- <<<

    // >>> Set grackle parameters <<<
    double redshift    = 2;
    double density     = 1e-24;
    double temperature = 1e5;

    int grid_rank = 3;
    int grid_dimensions[] = {grid_dim_i,100,100};
    // >>> ---------------------- <<<

    initialise_grackle_structs(primordial_chemistry, density, temperature, redshift, grid_rank,
                                grid_dimensions, &my_chemistry, &my_rates, &my_fields, &my_units);

    // >>> Setup timing framework <<<
    int num_timing_iter = 1000;
    // >>> ---------------------- <<<

    // >>> Benchmark pressure, temperature and gamma calculations <<<
    run_benchmark(&my_chemistry, &my_rates, &my_fields, &my_units, numTeams, nThreadsPerTeam,
                    num_timing_iter, timing_dir);
    // >>> ------------------------------------------------------ <<<

    // >>> Cleanup memory <<<
    free_grackle_structs(&my_chemistry, &my_rates, &my_fields, &my_units);
    // >>> -------------- <<<

    return 0;
}