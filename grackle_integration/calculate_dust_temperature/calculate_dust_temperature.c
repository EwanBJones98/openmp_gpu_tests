#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "../grackle_chemistry.h"
#include "../grackle_fields.h"
#include "../grackle_units.h"
#include "../grackle_rates.h"
#include "../grackle_utility.h"
#include "../physical_constants.h"

// This is a placeholder for the actual calculate temperature function
void calculate_temperature(double *temperature, int field_size,
                           double temperature_value)
{
    for (int i=0; i<field_size; i++)
    {
        temperature[i] = temperature_value;
    }
}

int calculate_dust_temperature(grackle_chemistry *my_chemistry,
                               grackle_rates *my_rates,
                               grackle_units *my_units,
                               grackle_fields *my_fields)
{
    if (!my_chemistry->use_grackle)
    {
        return 1;
    }

    if (my_chemistry->dust_chemistry < 1 && my_chemistry->h2_on_dust < 1)
    {
        return 1;
    }

    set_comoving_units(my_units);
    set_temperature_units(my_units);

    int field_size = 1;
    for (int dim=0; dim < my_fields->grid_rank; dim++)
    {
        field_size *= my_fields->grid_dimensions[dim];
    }

    // Set log values of start and end of lookup tables
    double log_temp_start, log_temp_end, d_log_temp;
    log_temp_start = log(my_chemistry->temperature_start);
    log_temp_end   = log(my_chemistry->temperature_end);
    d_log_temp     = (log_temp_end - log_temp_start) / 
                     (my_chemistry->number_of_temperature_bins - 1);
    
    // Set CMB temperature
    double redshift, temp_cmb;
    redshift = 1 / (my_units->a_value * my_units->a_units) - 1;
    temp_cmb = 2.73 * (1 + redshift);

    // Calculate cooling units
    double time_base, length_base, density_base, cool_units;
    time_base    = my_units->time_units;
    length_base  = my_units->co_length_units / (my_units->a_units * my_units->a_value);
    density_base = my_units->co_density_units *
                    pow(my_units->a_units * my_units->a_value, 3);
    cool_units   = (pow(my_units->a_value, 5) * pow(length_base, 2) * pow(mh, 2)) /
                    (pow(time_base, 3) * density_base);

    // >>>> DEFINITIONS OF LOCALS <<<<<
    bool iteration_mask[field_size];
    int index[field_size];
    double isrf[field_size], nH[field_size], gas_grain[field_size],
           gas_temp[field_size], log_gas_temp[field_size],
           log_gas_temp_interp_1[field_size],
           log_gas_temp_interp_2[field_size],
           log_gas_temp_interp_grad[field_size];
    // >>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<

    // All particles to be calculated
    for (int i=0; i<field_size; i++)
    {
        iteration_mask[i] = true;
    }

    // Loop over all particles to precompute interpolation
    for (int i=0; i<field_size; i++)
    {

        // Compute interstellar radiation field
        if (my_chemistry->use_isrf_field)
        {
            isrf[i] = my_fields->isrf_habing[i];
        } else {
            isrf[i] = my_chemistry->interstellar_radiation_field;
        }

        // Compute hydrogen number density
        nH[i] = my_fields->HI_density[i] + my_fields->HII_density[i];
        if (my_chemistry->primordial_chemistry > 1)
        {
            nH[i] += my_fields->H2I_density[i] + my_fields->H2II_density[i];
        }
        nH[i] *= my_units->co_density_units / mh; // Do not use dom_unit \
                                                     not converted to proper

        // Compute log temperature and truncate if above/below table max/min
        gas_temp[i] = my_fields->temperature[i];
        log_gas_temp[i] = log(gas_temp[i]);
        log_gas_temp[i] = max(log_gas_temp[i], log_temp_start);
        log_gas_temp[i] = min(log_gas_temp[i], log_temp_end);

        // Convert index into the table and precompute parts of the linear interpolation
        index[i] = (int) (log_gas_temp[i] - log_temp_start) / d_log_temp;
        index[i] = max(0, index[i]);
        index[i] = min(my_chemistry->number_of_temperature_bins - 1, index[i]);

        log_gas_temp_interp_1[i]    = log_temp_start + (index[i] - 1)*d_log_temp;
        log_gas_temp_interp_2[i]    = log_temp_start + (index[i]    )*d_log_temp;
        log_gas_temp_interp_grad[i] = (log_gas_temp[i] - log_gas_temp_interp_1[i]) /
                                        (log_gas_temp_interp_2[i] - log_gas_temp_interp_1[i]);

        // Look up values and do a linear temperature in log(T)
        // Convert back to cgs
        gas_grain[i] = my_rates->gas_grain[index[i]] + log_gas_temp_interp_grad[i] * 
                        (my_rates->gas_grain[index[i]+1] - my_rates->gas_grain[index[i]]);
        gas_grain[i] *= my_chemistry->local_dust_to_gas_ratio * cool_units / mh;
    }

    // Calculate dust temperature


    return 1;
}