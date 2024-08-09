#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "grackle_types.h"
#include "grackle_chemistry_data.h"
#include "write_timings.h"

int write_timings(int num_timing_iter, double *calc_times, double *data_times, int num_teams,
                   int num_threads_per_team, char mode[10], grackle_field_data *my_fields,
                   chemistry_data *my_chemistry, int new_file, char *filename)
{

    // Calculate mean and standard deviation of timing arrays
    double calc_mean=0., data_mean=0.;
    for (int i=0; i<num_timing_iter; i++)
    {
        calc_mean += calc_times[i] / num_timing_iter;
        data_mean += data_times[i] / num_timing_iter;
    }

    double calc_stdev=0., data_stdev=0.;
    for (int i=0; i<num_timing_iter; i++)
    {
        calc_stdev += pow(calc_times[i] - calc_mean, 2);
        data_stdev += pow(data_times[i] - data_mean, 2);
    }
    calc_stdev = sqrt(calc_stdev/num_timing_iter);
    data_stdev = sqrt(data_stdev/num_timing_iter);

    FILE *fptr;

    // Check to see if file exists. If it does not then a new one will have to be created.
    if (!(fptr = fopen(filename, "r"))) new_file=1;    

    if (new_file)
    {
        fptr = fopen(filename, "w+");
        if (fptr == NULL) return 0;

        fprintf(fptr, "Mode      | Primordial chemistry | Fields i dimension | Fields j dimension | Fields k dimension | Number of teams | Number of threads per team | Mean calculation time (s) | Standard deviation (s) | Mean data time (s) | Standard deviation (s)\n");
        fprintf(fptr, "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    } else {
        
        fptr = fopen(filename, "a");
        if (fptr == NULL) return 0;
    }

    fprintf(fptr, "%-10s|%-22d|%-20d|%-20d|%-20d|%-17d|%-28d|%-27g|%-24g|%-20g|%-23g\n", mode, my_chemistry->primordial_chemistry, my_fields->grid_dimension[0],
                    my_fields->grid_dimension[1], my_fields->grid_dimension[2], num_teams, num_threads_per_team, calc_mean, calc_stdev, data_mean, data_stdev);

    fclose(fptr);

    return 1;
}