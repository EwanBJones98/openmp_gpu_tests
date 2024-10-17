#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "grackle_types.h"
#include "grackle_chemistry_data.h"
#include "write_timings.h"

int write_timings(int num_timing_iter, double *times, int num_teams,
                   int num_threads_per_team, char mode[10], grackle_field_data *my_fields,
                   chemistry_data *my_chemistry, int new_file, char *filename)
{

    // Calculate mean and standard deviation of timing arrays
    double mean=0;
    for (int i=0; i<num_timing_iter; i++)
        mean += times[i] / num_timing_iter;

    double stdev=0;
    for (int i=0; i<num_timing_iter; i++)
        stdev += pow(times[i] - mean, 2);
    stdev = sqrt(stdev/num_timing_iter);

    FILE *fptr;
    // Check to see if file exists. If it does not then a new one will have to be created.
    if (!(fptr = fopen(filename, "r"))) new_file=1;    

    if (new_file)
    {
        fptr = fopen(filename, "w+");
        if (fptr == NULL) return 0;

        fprintf(fptr, "Mode      | Primordial chemistry | Fields i dimension | Fields j dimension | Fields k dimension | Number of teams | Number of threads per team | Mean time (s) | Standard deviation (s)\n");
        fprintf(fptr, "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    } else {
        
        fptr = fopen(filename, "a");
        if (fptr == NULL) return 0;
    }

    fprintf(fptr, "%-10s|%-22d|%-20d|%-20d|%-20d|%-17d|%-28d|%-15g|%-23g\n", mode, my_chemistry->primordial_chemistry, my_fields->grid_dimension[0],
                    my_fields->grid_dimension[1], my_fields->grid_dimension[2], num_teams, num_threads_per_team, mean, stdev);

    fclose(fptr);

    return 1;
}