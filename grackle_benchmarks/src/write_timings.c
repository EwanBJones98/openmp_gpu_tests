#include <stdlib.h>
#include <stdio.h>
#include "grackle_types.h"
#include "grackle_chemistry_data.h"

int write_timings(int num_iterations, double mean_time, double stdev, int num_teams,
                    int num_threads, char mode[10], grackle_field_data *my_fields,
                    chemistry_data *my_chemistry, int new_file, char filename[100])
{

    FILE *fptr;

    // Check to see if file exists. If it does not then a new one will have to be created.
    if (!(fptr = fopen(filename, "r"))) new_file=1;    

    if (new_file)
    {
        fptr = fopen(filename, "w+");
        if (fptr == NULL) return 0;

        fprintf(fptr, "Mode | Primordial chemistry | Fields i dimension | Fields j dimension | Fields k dimension | Number of timing iterations | Number of teams | Number of threads | Number of threads per team | Mean time (s) | Standard deviation (s)\n");
        fprintf(fptr, "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    } else {
        
        fptr = fopen(filename, "a");
        if (fptr == NULL) return 0;
    }

    fprintf(fptr, "%-5s|%-22d|%-20d|%-20d|%-20d|%-29d|%-17d|%-19d|%-28d|%-15g|%-23g\n",
                    mode, my_chemistry->primordial_chemistry, my_fields->grid_dimension[0],
                    my_fields->grid_dimension[1], my_fields->grid_dimension[2],
                    num_iterations, num_teams, num_threads, num_threads/num_teams,
                    mean_time, stdev);

    fclose(fptr);
}