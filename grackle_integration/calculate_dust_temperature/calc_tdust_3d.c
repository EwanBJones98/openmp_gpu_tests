#include "../grackle_fields.h"
#include "../grackle_chemistry.h"
#include "../grackle_units.h"
#include "../grackle_rates.h"

#include <math.h>

// Function to calculate the dust temperature
int calc_tdust_3d(grackle_chemistry *my_chemistry,
                  grackle_fields *my_fields,
                  grackle_rates *my_rates,
                  grackle_units *my_units)
{
    /*
        >>>> COMPUTE THE DUST TEMPERATURE <<<<

        written by: Britton Smith
        date: July, 2011

        ported by: Ewan Jones (Fortran->C)
        date: May, 2024

        PURPOSE:
            Calculate dust heat balance to get the dust temperature.

        INPUTS:
            my_chemistry - pointer to grackle_chemistry struct
            my_fields    - pointer to grackle_fields struct
            my_rates     - pointer to grackle_rates struct
            my_units     - pointer to grackle_units struct
    */

    // Set log values of start and end of lookup tables
    double log_temp_start, log_temp_end, d_log_temp;
    log_temp_start = log(my_chemistry->temperature_start);
    log_temp_end   = log(my_chemistry->temperature_end);
    d_log_temp     = (log_temp_end - log_temp_start) / (my_chemistry->number_of_temperature_bins - 1);

    // Set units
    double time_base, length_base, density_base, cool_unit, z;
    time_base = my_units->time_units;
    length_base = my_units->length_units;

    



}