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

#include "write_timings.h"
#include "calculate_pressure_gpu.h"

#ifdef _OPENMP
    #include <omp.h>
#else
    #include <time.h>
    static int omp_get_thread_num() {return 0;}
    static int omp_get_num_threads() {return 1;}
    static double omp_get_wtime() {return (clock() / CLOCKS_PER_SEC);}
#endif


double tiny_number = 1e-20;

int calculate_pressure_gpu(chemistry_data *my_chemistry,
                           chemistry_data_storage *my_rates,
                           code_units *my_units,
                           grackle_field_data *my_fields,
                           gr_float *pressure,
                           double pressure_units,
                           int num_iter, double *timings_calc,
                           double *timings_data,
                           double *timings_oneloop,
                           int nteams, int nthreadsPerTeam)
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
    double start_time_data;
    start_time_data = omp_get_wtime();
    #if defined(_OPENMP) && defined(GPU)   
        enter_calculate_pressure(pressure, temperature_units, GammaInverse,
                                tiny_number, my_chemistry, my_fields,
                                my_units, &ind_helper);
    #endif
    *timings_data = omp_get_wtime() - start_time_data;

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
    #ifdef LOOP_STRUCTURE_TEST
        double start_time_oneloop;
    #endif
    for (int iter=0; iter<num_iter; iter++)
    {

        #ifdef LOOP_STRUCTURE_TEST

            start_time_oneloop = omp_get_wtime();

            #if defined(_OPENMP) && defined(CPU)
                #pragma omp parallel for schedule(runtime)
            #elif defined(_OPENMP) && defined(GPU)
                #ifdef WORKSHARING_TEST
                    #pragma omp target teams distribute parallel for num_teams(nteams) num_threads(nthreadsPerTeam)
                #else
                    #pragma omp target teams distribute parallel for
                #endif
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
            } // End loop


            timings_oneloop[iter] = omp_get_wtime() - start_time_oneloop;

            if (iter == 0) start_time_data = omp_get_wtime();
            #if defined(_OPENMP) && defined(GPU)
                #pragma omp target update from(pressure[:field_length])
            #endif
            if (iter == 0) *timings_data += start_time_data - omp_get_wtime();

            // Check that the pressure has been updated and reset for next loop
            for (int i=0; i<field_length; i++)
            {
                if (pressure[i] == -1) return 0;

                pressure[i] = -1;
            }

            #if defined(_OPENMP) && defined(GPU)
                #pragma omp target update to(pressure[:field_length])
            #endif

        #else // end loop test

            start_time = omp_get_wtime();

            #if defined(_OPENMP) && defined(CPU)
                #pragma omp parallel for schedule(runtime)
            #elif defined(_OPENMP) && defined(GPU)
                #ifdef WORKSHARING_TEST
                    #pragma omp target teams distribute parallel for num_teams(nteams) num_threads(nthreadsPerTeam)
                #else
                    #pragma omp target teams distribute parallel for num_teams(nteams) num_threads(nthreadsPerTeam)
                #endif
            #endif
            for (int index=0; index<field_length; index++)
            {   
                #ifdef USE_BUFFER_ZONES
                    if (buffer_zones[index] == 1) continue;
                #endif

                pressure[index] = (my_chemistry->Gamma -1.0) * my_fields->density[index]
                                    * my_fields->internal_energy[index];
            }

            if (my_chemistry->primordial_chemistry > 1)
            {

                #if defined(_OPENMP) && defined(CPU)
                    #pragma omp parallel for schedule(runtime)
                #elif defined(_OPENMP) && defined(GPU)
                    #ifdef WORKSHARING_TEST
                        #pragma omp target teams distribute parallel for num_teams(nteams) num_threads(nthreadsPerTeam)
                    #else
                        #pragma omp target teams distribute parallel for num_teams(nteams) num_threads(nthreadsPerTeam)
                    #endif
                #endif
                for (int index=0; index<field_length; index++)
                {

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
                } // End loop
            } // End if primordial_chemistry > 1

            timings_calc[iter] = omp_get_wtime() - start_time;
            
            if (iter == 0) start_time_data = omp_get_wtime();
            #if defined(_OPENMP) && defined(GPU)
                #pragma omp target update from(pressure[:field_length])
            #endif
            if (iter == 0) *timings_data += omp_get_wtime() - start_time_data;

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

        start_time_data = omp_get_wtime();
        #if defined(_OPENMP) && defined(GPU)
            exit_calculate_pressure(pressure, temperature_units, GammaInverse,
                                    tiny_number, my_chemistry, my_fields,
                                    my_units, &ind_helper);
        #endif
        *timings_data += omp_get_wtime() - start_time_data;

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

/*
    Main function accepts command line arguments based on the compile macro used.
    Below are a list of command line arguments for each. The default set should
    be used if the macro is not specifically listed below.

        + WORKSHARING_TEST:
            primordial_chemistry, nteams_start, nteams_stride, nteams_numpoints
                nthreads_start, nthreads_stride, nthreads_numpoints,
                output_filepath

            NOTE: The grid dimensions are set in the code when this macro is enabled, please
                    change them below.

            NOTE: nthreads that you pass in as command line arguments refers to the number of threads
                    per team.

            NOTE: values are log-spaced between the start and end points

        + other:
            primordial_chemistry, grid_dimensions[0], grid_dimensions[1],
                grid_dimensions[2], output_filepath

*/

int main(int argc, char *argv[])
{

    #ifdef WORKSHARING_TEST
        fprintf(stdout, "WORKSHARING_TEST = active\n");
    #else
        fprintf(stdout, "WORKSHARING_TEST = inactive\n");
    #endif


    #ifdef PRESET_TESTS
        fprintf(stdout, "PRESET_TESTS = active\n");
    #else
        fprintf(stdout, "PRESET_TESTS = inactive\n");
    #endif

    int grid_rank = 3;

    char filepath[200] = "/home/ejones/codes/openmp_gpu_tests/grackle_benchmarks/";
    int primordial_chemistry;
    int grid_dimensions[grid_rank];

    char *str_buffer;
    #ifdef WORKSHARING_TEST
        int nteams_start, nteams_end, nteams_npoints,
            nthreads_start, nthreads_end, nthreads_npoints;

        if (argc != 9)
        {
            fprintf(stderr, "Wrong number of command line arguments!\nExpected 8 (excluding filename), received %d\n", argc);
            exit(0);
        }

        primordial_chemistry = (int) strtol(argv[1], &str_buffer, 10);
        nteams_start         = (int) strtol(argv[2], &str_buffer, 10);
        nteams_end           = (int) strtol(argv[3], &str_buffer, 10);
        nteams_npoints       = (int) strtol(argv[4], &str_buffer, 10);
        nthreads_start       = (int) strtol(argv[5], &str_buffer, 10);
        nthreads_end         = (int) strtol(argv[6], &str_buffer, 10);
        nthreads_npoints     = (int) strtol(argv[7], &str_buffer, 10);
        
        strcat(filepath, argv[8]);

        int num_teams_logspace[nteams_npoints], num_threads_logspace[nthreads_npoints];
        double logdelta_teams, logdelta_threads;
        logdelta_teams   = (log10(nteams_end) - log10(nteams_start)) / (nteams_npoints - 1);
        logdelta_threads = (log10(nthreads_end) - log10(nthreads_start)) / (nthreads_npoints - 1);
        for (int i=0; i < nteams_npoints; i++)
            num_teams_logspace[i] = (int) pow(10, log10(nteams_start) + i*logdelta_teams);
        for (int i=0; i < nthreads_npoints; i++)
            num_threads_logspace[i] = (int) pow(10, log10(nthreads_start) + i*logdelta_threads);

    #elif PRESET_TESTS

        if (argc != 3)
        {
            fprintf(stderr, "Wrong number of command line arguments! Expected 2, received %d\n", argc);
            exit(0);
        }

        primordial_chemistry = (int) strtol(argv[1], &str_buffer, 10);

        strcat(filepath, argv[2]);


        // int len_preset_tests_idim = 4;
        // int preset_tests_idim[]   = {100,1000,3000,4000};
        int len_preset_tests_idim = 1;
        int preset_tests_idim[]   = {4000};
        int preset_tests_jkdim    = 100;

        // int len_preset_tests_threadsPerTeam = 3;
        // int preset_tests_threadsPerTeam[]   = {256,512,-1}; // -1 means use default value
        int len_preset_tests_threadsPerTeam = 1;
        int preset_tests_threadsPerTeam[] = {-1};

    #else

        if (argc != 6) {
            fprintf(stderr, "Wrong number of command line arguments!\nExpected 5, received %d\n", argc);
            exit(0);
        }

        primordial_chemistry = (int) strtol(argv[1], &str_buffer, 10);
        grid_dimensions[0]   = (int) strtol(argv[2], &str_buffer, 10);
        grid_dimensions[1]   = (int) strtol(argv[3], &str_buffer, 10);
        grid_dimensions[2]   = (int) strtol(argv[4], &str_buffer, 10);
        
        strcat(filepath, argv[5]);
    #endif

    chemistry_data my_chemistry;
    chemistry_data_storage my_rates;
    code_units my_units;
    grackle_field_data my_fields;

    // Details of worker093
    int max_threads_per_team = 1024;
    int max_threads_per_sm   = 1536;
    int num_sm               = 142;

    //! >>> USER SETTINGS GPU <<<
    int num_threads_per_team = 512;
    int num_teams            = num_sm * max_threads_per_sm / num_threads_per_team;
    //! >>> ----------------- <<<

    int num_threads          = num_threads_per_team * num_teams;

    //! >>> USER SETTINGS <<<
    double initial_redshift = 2;
    double temperature = 1000;

    int num_timing_iter = 1000;

    int default_gpu_worksharing = 1;
    int limit_max_threads       = 1; // If set to 1 then the maximum number of threads allocated by default is scaled down to the hardware maximum

    int overwrite=0;

    #ifdef WORKSHARING_TEST
        grid_dimensions[0] = 1000;
        grid_dimensions[1] = 100;
        grid_dimensions[2] = 100;
    #endif
    //! >>> ---------- <<<

    // >>> Loop over worksharing distribution in preset tests <<<
    #ifdef PRESET_TESTS
    for (int i_pt=0; i_pt<len_preset_tests_idim; i_pt++){
        for (int j_pt=0; j_pt<len_preset_tests_threadsPerTeam; j_pt++){

            if (preset_tests_threadsPerTeam[j_pt] == -1){
                default_gpu_worksharing = 1;
            } else {
                num_threads_per_team = preset_tests_threadsPerTeam[j_pt];
                num_teams            = num_sm * max_threads_per_sm / num_threads_per_team;
                num_threads          = num_threads_per_team * num_teams;

                default_gpu_worksharing = 0;
            }
            
            grid_dimensions[0] = preset_tests_idim[i_pt];
            grid_dimensions[1] = preset_tests_jkdim;
            grid_dimensions[2] = preset_tests_jkdim; 
    #endif

    // >>> Get parallelism info <<<
    
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
            omp_gpu_info(&gpu_active, &num_threads, &num_teams, default_gpu_worksharing);
            if (limit_max_threads == 1)
            {
                int num_threads_per_team_default = num_threads / num_teams;
                if (num_threads > max_threads_per_sm * num_sm)
                {
                    num_threads = max_threads_per_sm * num_sm;
                    num_teams   = num_threads / num_threads_per_team_default;

                    omp_gpu_info(&gpu_active, &num_threads, &num_teams, 0);
                }
            }

            if (gpu_active == 0)
            {
                fprintf(stdout, "Running with GPU parallelism!\n\tnumber of threads: %d\n\tnumber of teams:%d\n\n",
                            num_threads, num_teams);
            } else {
                fprintf(stdout, "GPU inactive! Exiting...\n");
            }
        #endif
    #else
        num_threads = 1;
        num_teams   = 1;
        fprintf(stdout, "Running in serial on the CPU!\n\n");
        strcpy(mode, "serial");
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

    double timings[num_timing_iter], *timings_oneloop, data_transfer_time;
    #if defined(LOOP_STRUCTURE_TEST)
        timings_oneloop = malloc(sizeof(double) * num_timing_iter);
    #elif defined(WORKSHARING_TEST)
        int nteams_req, nthreads_req;

        for (int team_index=0; team_index<nteams_npoints; team_index++)
        {
            nteams_req = num_teams_logspace[team_index];
            for (int thread_index=0; thread_index<nthreads_npoints; thread_index++)
            {
                nthreads_req = num_threads_logspace[thread_index];

                fprintf(stdout, "requested: teams = %d, threads = %d\n", nteams_req, nthreads_req);

                // if (!omp_gpu_check_worksharing_config(&nteams_req, &nthreads_req))
                    // continue;

                num_teams   = nteams_req;
                num_threads = nteams_req * nthreads_req;

                fprintf(stdout, "nteams=%d, nthreads=%d\n", num_teams, num_threads);
                fprintf(stdout, "nteams requested=%d, nthreads requested=%d\n", nteams_req, nthreads_req);
                fflush(stdout);
    #endif

    calculate_pressure_gpu(&my_chemistry, &my_rates, &my_units, &my_fields,
                           pressure, pressure_units, num_timing_iter, timings,
                           &data_transfer_time, timings_oneloop, num_teams,
                           num_threads/num_teams);


    double avg_time = 0, stdev=0;
    for (int i=0; i<num_timing_iter; i++)
        avg_time += timings[i] / num_timing_iter;
    for (int i=0; i<num_timing_iter; i++)
        stdev += pow(timings[i] - avg_time, 2);
    stdev = sqrt(stdev/num_timing_iter);

    #ifdef LOOP_STRUCTURE_TEST
        double avg_time_oneloop = 0, stdev_oneloop=0;
        for (int i=0; i<num_timing_iter; i++)
            avg_time_oneloop += timings_oneloop[i] / num_timing_iter;
        for (int i=0; i<num_timing_iter; i++)
            stdev_oneloop += pow(timings_oneloop[i] - avg_time, 2);
        stdev_oneloop = sqrt(stdev_oneloop/num_timing_iter);
    #endif


    #if defined(OPT_TEST)
        char opt_flag[10];
        strcpy(opt_flag, OPT_TEST);
        write_timings_optimization(opt_flag, num_timing_iter, avg_time, stdev,
                                    num_teams, num_threads, mode, &my_fields,
                                    &my_chemistry, overwrite, filepath);
    #elif defined(LOOP_STRUCTURE_TEST)
        write_timings_loop_structure_test(num_timing_iter, avg_time_oneloop, stdev_oneloop,
                                            avg_time, stdev, num_teams, num_threads, mode,
                                            &my_fields, &my_chemistry, overwrite, filepath);
    #else
        write_timings(num_timing_iter, avg_time, stdev, num_teams, num_threads,
                        mode, &my_fields, &my_chemistry, overwrite, filepath);
    #endif

    local_free_chemistry_data(&my_chemistry, &my_rates);
    free_grackle_fields(&my_fields);

    #ifdef WORKSHARING_TEST
            }
        }
    #endif

    #ifdef PRESET_TESTS
        }
    }
    #endif
        
    return 1;
}