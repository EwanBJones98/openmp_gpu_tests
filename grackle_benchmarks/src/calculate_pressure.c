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

#include "calculate_pressure.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

void calculate_pressure(double *pressure,
                        int field_length,
                        chemistry_data *my_chemistry,
                        chemistry_data_storage *my_rates,
                        grackle_field_data *my_fields,
                        code_units *my_units,
                        int nTeams,
                        int nThreadsPerTeam)
{
    
    #if defined(_OPENMP) && defined(CPU)
        #pragma omp parallel for schedule(runtime)
    #elif defined(_OPENMP) && defined(GPU)
        #pragma omp target teams distribute parallel for num_teams(nTeams) num_threads(nThreadsPerTeam)
    #endif
    for (int index=0; index<field_length; index++)
    {   
        pressure[index] = (my_chemistry->Gamma - 1.0) * my_fields->density[index]
                            * my_fields->internal_energy[index];
    }

    if (my_chemistry->primordial_chemistry > 1)
    {

        #if defined(_OPENMP) && defined(CPU)
            #pragma omp parallel for schedule(runtime)
        #elif defined(_OPENMP) && defined(GPU)
            #pragma omp target teams distribute parallel for num_teams(nTeams) num_threads(nThreadsPerTeam)
        #endif
        for (int index=0; index<field_length; index++)
        {
            double number_density = 0.25 * (my_fields->HeI_density[index] + my_fields->HeII_density[index]
                                                + my_fields->HeIII_density[index])
                                        + my_fields->HI_density[index] + my_fields->HII_density[index]
                                        + my_fields->HM_density[index] + my_fields->e_density[index];

            double nH2 = 0.5 * (my_fields->H2I_density[index] + my_fields->H2II_density[index]);

            // First, approximate temperature
            if (number_density == 0) number_density = 1e-20;
            double temp = max(my_units->temperature_units * pressure[index] / (number_density + nH2), 1);

            // Only do full computation if there is a reasonable amount of H2.
            // The second term in GammaH2Inverse accounts for the vibrational
            //  degrees of freedom.
            double GammaInverse = 1.0 / (my_chemistry->Gamma - 1.0);
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
    
            pressure[index] = max(1e-20, pressure[index]);
        } // End loop
    } // End if primordial_chemistry > 1
}