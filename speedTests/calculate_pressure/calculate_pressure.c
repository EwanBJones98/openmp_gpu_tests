#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
    #include <omp.h>
#else
    static int omp_get_thread_num() {return 0;}
    static int omp_get_num_threads() {return 1;}
    static double omp_get_wtime() {return (clock() / CLOCKS_PER_SEC);}
#endif

#ifndef TINY_NUMBER
    #define TINY_NUMBER 1e-20
#endif

typedef struct
{
    int num_iterations;

    double *all_elapsed_times;
    double *all_calculation_times;

    double avg_elapsed_time;
    double avg_calculation_time;
} timings;

typedef struct
{
    int num_teams;
    int num_threads;
    int num_threads_per_team;

    char mode[10];
} omp_settings;

typedef struct
{
    int n_dims;
    int* grid_dims;
    int num_parts;

    double *density;
    double *internal_energy;
    double *pressure;
} grackle_fields;

typedef struct
{
    double gamma;
} grackle_chemistry;

void free_arrays(grackle_fields *my_fields, timings *my_timings)
{
    free(my_fields->density);
    free(my_fields->internal_energy);
    free(my_fields->pressure);

    free(my_timings->all_calculation_times);
    free(my_timings->all_elapsed_times);
}

void copy_array(double* destination_array, double* source_array, int array_length)
{
    for (int i=0; i < array_length; i++)
    {
        destination_array[i] = source_array[i];
    }
}

void get_omp_settings(omp_settings *my_omp, int number_of_teams)
{
    // Get number of threads being used.
    char mode[10];
    int n_threads_per_team, n_teams, n_threads;
    #ifdef CPU
        strcpy(my_omp->mode, "cpu");
        #pragma omp parallel
        {
            #pragma omp single
            {
                n_threads          = omp_get_num_threads();
                n_teams            = -1;
                n_threads_per_team = -1;
            }
        }
    #elif GPU
        strcpy(my_omp->mode, "gpu");
        #pragma omp target teams map(tofrom:n_threads,n_teams,n_threads_per_team)\
                num_teams(number_of_teams)
        {
            #pragma omp parallel
            {
            #pragma omp single
            {
            n_threads_per_team = omp_get_num_threads();
            n_teams = omp_get_num_teams();
            n_threads = n_threads_per_team * n_teams;
            }
            }
        }
    #elif SERIAL
        strcpy(my_omp->mode, "serial");
        n_threads          = omp_get_num_threads();
        n_teams            = -1;
        n_threads_per_team = -1;
    #endif

    my_omp->num_threads          = n_threads;
    my_omp->num_teams            = n_teams;
    my_omp->num_threads_per_team = n_threads_per_team;
}

void allocate_memory_fields(grackle_fields *my_fields, int grid_dims[])
{
    int num_parts = grid_dims[0];
    for (int i=1; i < 3; i++)
    {
            num_parts *= grid_dims[i];
    }

    my_fields->n_dims    = 3;
    my_fields->num_parts = num_parts;
    my_fields->grid_dims = grid_dims;

    my_fields->density         = (double*) malloc(sizeof(double) * num_parts);
    my_fields->internal_energy = (double*) malloc(sizeof(double) * num_parts);
    my_fields->pressure        = (double*) malloc(sizeof(double) * num_parts);
}

void allocate_memory_timings(timings *my_timings, int num_iterations)
{
    my_timings->num_iterations = num_iterations;

    my_timings->all_elapsed_times     = (double*) malloc(sizeof(double) * num_iterations);
    my_timings->all_calculation_times = (double*) malloc(sizeof(double) * num_iterations);
}

void initialise_grackle_fields(grackle_fields *my_fields, int field_dimensions, int dimension_lengths[])
{
    for (int i=0; i < my_fields->num_parts; i++)
    {
        my_fields->density[i]         = 1e5;
        my_fields->internal_energy[i] = 1e-5;
        my_fields->pressure[i]        = TINY_NUMBER;
    }
}

void calculate_pressure(grackle_fields *my_fields, grackle_chemistry *my_chemistry,
                        omp_settings *my_omp, timings *my_timings, int calc_index)
{

    double time_start_total, time_end_total;
    double time_start_calc, time_end_calc;

    time_start_total = omp_get_wtime();

    // Unpacking the structs like this is stupid but must be done to pass arrays to the device when
    //  using GPU parallelism
    double *_density, *_internal_energy, *_pressure;
    _density         = malloc(sizeof(double) * my_fields->num_parts);
    _internal_energy = malloc(sizeof(double) * my_fields->num_parts);
    _pressure        = malloc(sizeof(double) * my_fields->num_parts);

    copy_array(_density, my_fields->density, my_fields->num_parts);
    copy_array(_internal_energy, my_fields->internal_energy, my_fields->num_parts);
    copy_array(_pressure, my_fields->pressure, my_fields->num_parts);

    double _gamma;
    int _num_parts, _num_teams;
    _gamma     = my_chemistry->gamma;
    _num_parts = my_fields->num_parts;
    _num_teams = my_omp->num_teams;

    #ifdef GPU
        #pragma omp target enter data map(to:_density[:_num_parts], _internal_energy[:_num_parts],\
                _pressure[:_num_parts], _gamma, _num_parts, _num_teams)
    #endif

    time_start_calc = omp_get_wtime();

    #ifdef GPU
        #pragma omp target teams distribute parallel for num_teams(_num_teams)
    #elif CPU
        #pragma omp parallel for
    #endif

    for (int i=0; i < _num_parts; i++)
    {
        _pressure[i] = (_gamma - 1.) * _density[i] * _internal_energy[i];

        if (_pressure[i] < TINY_NUMBER) _pressure[i] = TINY_NUMBER;
    }

    time_end_calc = omp_get_wtime();

    #ifdef GPU
        #pragma omp target exit data map(from: _pressure[:_num_parts]) map(delete:_gamma, _density[:_num_parts],\
                _internal_energy[:_num_parts], _num_parts, _num_teams)
    #endif

    // Place modified arrays back into structs
    copy_array(my_fields->pressure, _pressure, my_fields->num_parts);

    free(_density);
    free(_internal_energy);
    free(_pressure);

    time_end_total = omp_get_wtime();
    my_timings->all_calculation_times[calc_index] = time_end_calc - time_start_calc;
    my_timings->all_elapsed_times[calc_index]     = time_end_total - time_start_total;
}

void write_timings(timings *my_timings, omp_settings *my_omp, int num_parts)
{
    double sum_calc_time = 0., sum_elapsed_time = 0.;
    for (int iter=0; iter<my_timings->num_iterations; iter++)
    {
        sum_calc_time    += my_timings->all_calculation_times[iter];
        sum_elapsed_time += my_timings->all_elapsed_times[iter];
    }
    my_timings->avg_calculation_time = sum_calc_time / my_timings->num_iterations;
    my_timings->avg_elapsed_time     = sum_elapsed_time / my_timings->num_iterations;

    FILE *fptr;

    //Check if file exists and create if it does not
    if ((fptr = fopen("timings.txt", "r")) == NULL)
    {
        if ((fptr = fopen("timings.txt", "w+")) == NULL)
        {
            fprintf(stderr, "Failed to create timings.txt file. Exiting...\n");
            exit(3);
        }
        else
        {
            fprintf(fptr, "| %-11s | %-7s | %-9s | %-18s | %-11s | %-14s | %-28s | %-22s |\n", "OpenMP mode", "# teams",
                        "# threads", "# threads per team", "# particles", "# calculations", "average calculation time (s)",
                        "average total time (s)");
        }
    }
    fclose(fptr);

    // Write to file
    if ((fptr = fopen("timings.txt", "a")) == NULL){
        fprintf(stderr, "Error opening timing file!\n");
        exit(4);
    }

    fprintf(fptr, "| %-11s | %-7d | %-9d | %-18d | %-11d | %-14d | %-28e | %-22e |\n", my_omp->mode, my_omp->num_teams, my_omp->num_threads,
            my_omp->num_threads_per_team, num_parts, my_timings->num_iterations, my_timings->avg_calculation_time,
            my_timings->avg_elapsed_time);

    fclose(fptr);
}

void run_test(grackle_fields *my_fields, timings *my_timings, int grid_dims[3], int num_teams)
{
    omp_settings my_omp;
    get_omp_settings(&my_omp, num_teams);

    int n_dims = 3;
    initialise_grackle_fields(my_fields, n_dims, grid_dims);

    grackle_chemistry my_chemistry;
    my_chemistry.gamma = 5/3.;

    for (int iter=0; iter<my_timings->num_iterations; iter++)
    {
     calculate_pressure(my_fields, &my_chemistry, &my_omp, my_timings, iter);
    }

    write_timings(my_timings, &my_omp, my_fields->num_parts);
}

int check_pressure(double correct_value, double *array_to_check, int array_length)
{
    for (int i=0; i<array_length; i++)
    {
        if (correct_value != array_to_check[i])
        {
            return 0;
        }
    }
    return 1;
}

int main(char argc, char *argv[])
{
    grackle_fields my_fields;
    int grid_dims[] = {100, 100, 100};
    allocate_memory_fields(&my_fields, grid_dims);

    timings my_timings;
    int n_iter = 100;
    allocate_memory_timings(&my_timings, n_iter);

    for (int n_teams=1; n_teams<=1000; n_teams++)
    {
        fprintf(stdout, "%d\n", n_teams);
        run_test(&my_fields, &my_timings, grid_dims, n_teams);
    }

    double result = (5/3. - 1.) * 1e-5 * 1e5;
    if (!check_pressure(result, my_fields.pressure, my_fields.num_parts))
    {
        fprintf(stdout, "Pressure calculation failed!\n");
        fprintf(stdout, "Expected result = %e\n", result);
        fprintf(stdout, "First five elements of calculated array were:\n");
        for (int i=0; i<5; i++)
        {
            fprintf(stdout, "\t pressure[%d] = %f\n", i, my_fields.pressure[i]);
        }
    }

    free_arrays(&my_fields, &my_timings);

    return 1;
}