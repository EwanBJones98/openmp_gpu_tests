#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "grackle.h"
#include "grackle_types.h"
#include "grackle_chemistry_data.h"
#include "phys_constants.h"
#include "grackle_macros.h"

#include "utility.h"

#include "calculate_temperature.h"
#include "calculate_pressure.h"

#ifdef _OPENMP
    #import <omp.h>
#endif

int calculate_temperature(double *temperature,
                          int field_length,
                          chemistry_data *my_chemistry,
                          chemistry_data_storage *my_rates,
                          grackle_field_data *my_fields,
                          code_units *my_units,
                          int nTeams,
                          int nThreadsPerTeam)
{
    // >>> First calculate the pressure <<<
    //  The pressure is stored in the temperature array to save memory
    calculate_pressure(temperature, field_length, my_chemistry, my_rates, my_fields, my_units, nTeams, nThreadsPerTeam);
    // >>> ---------------------------- <<<

    if (my_chemistry->primordial_chemistry == 0)
    {
        fprintf(stderr, "The tabulated solver used for primordial chemistry < 1 is not yet implemented within the GPU test suite.\n");
        return 1;
    }

    if (my_chemistry->primordial_chemistry > 0)
    {

        #if defined(CPU) && defined(_OPENMP)
            #pragma omp parallel for schedule(runtime)
        #elif defined(GPU) && defined(_OPENMP)
            #pragma omp target teams distribute parallel for num_teams(nTeams) num_threads(nThreadsPerTeam)
        #endif
        for (int i=0; i<field_length; i++)
        {
            double number_density = 0.25 * (my_fields->HeI_density[i] + my_fields->HeII_density[i] + 
                                            my_fields->HeIII_density[i]) + 
                                    my_fields->HI_density[i] + my_fields->HII_density[i] + 
                                    my_fields->e_density[i];
        

            //! Kinda sucks that the following if statements are inside the loop
            //!  but its probably better than copying arrays back and forth

            // Add H2            
            if (my_chemistry->primordial_chemistry > 1)
            {
                number_density += my_fields->HM_density[i] +
                0.5 * (my_fields->H2I_density[i] + my_fields->H2II_density[i]);
            }

            //! Do not worry about this for now
            // Metals
            // if (my_fields->metal_density != NULL)
            // {   
            //     number_density += my_fields->metal_density[i] * 1. / 16.;
            // }

            //Ignore deuterium

            temperature[i] *= my_units->temperature_units / max(number_density, 1e-20);
            temperature[i]  = max(temperature[i], 1);
        } // End for loop
    } // End if primordial_chemistry > 0

    return 1;
}