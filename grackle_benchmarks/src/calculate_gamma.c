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

#include "calculate_gamma.h"
#include "calculate_temperature.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

// This is the function that calculates gamma
int calculate_gamma(double *gamma,
                    int field_length,
                    chemistry_data *my_chemistry,
                    chemistry_data_storage *my_rates,
                    grackle_field_data *my_fields,
                    code_units *my_units,
                    int nTeams,
                    int nThreadsPerTeam)
{

    // If molecular hydrogen is not being used, just use monotonic.
    if (my_chemistry->primordial_chemistry < 2)
    {
        #if defined(CPU) && defined(_OPENMP)
            #pragma omp parallel for schedule(runtime)
        #elif defined(GPU) && defined(_OPENMP)
            #pragma omp target teams distribute parallel for num_teams(nTeams) num_threads(nThreadsPerTeam)
        #endif
        for (int i=0; i<field_length; i++)
            gamma[i] = my_chemistry->Gamma;
    } else {
        // If molecular hydrogen is being used then its more sophisticated.

        // Here we calculate the temperature values but store them in the gamma array to save memory
        if (!calculate_temperature(gamma, field_length, my_chemistry,
                                    my_rates, my_fields, my_units, nTeams, nThreadsPerTeam))
        {
            fprintf(stderr, "Error in calculate_temperature_gpu.\n");
            return 0;
        }

        /* Compute Gamma with molecular Hydrogen formula from Omukau \& Nishi
        astro-ph/9811308. */
        #if defined(_OPENMP) && defined(CPU)
            #pragma omp parallel for schedule (runtime)
        #elif defined(_OPENMP) && defined(GPU)
            #pragma omp target teams distribute parallel for num_teams(nTeams) num_threads(nThreadsPerTeam)
        #endif
        for (int i=0; i<field_length; i++)
        {
            double gamma_inverse = 1 / (my_chemistry->Gamma - 1.);

            double number_density = 0.25 * ( my_fields->HeI_density[i] + 
                                             my_fields->HeII_density[i] + 
                                             my_fields->HeIII_density[i] ) +
                                     my_fields->HI_density[i] +
                                     my_fields->HII_density[i] +
                                     my_fields->HM_density[i] +
                                     my_fields->e_density[i];

            double nH2 = 0.5 * (my_fields->H2I_density[i] + my_fields->H2II_density[i]);

            /* Only do full computation if there is a reasonable amount of H2.
            The second term in GammaH2Inverse accounts for the vibrational
            degrees of freedom. */

            double GammaH2Inverse = 0.5*5.0;
            if (nH2 / number_density > 1e-3)
            {
                double x = 6100.0 / gamma[i]; // Gamma array here is currently holding temperature values
                if (x < 10.0)
                    GammaH2Inverse = 0.5 * (5. + 2. * x*x * exp(x)/POW(exp(x)-1., 2));
            }

            // Add in H2
            gamma[i] = 1. + (nH2 + number_density) / (nH2 * GammaH2Inverse + number_density * gamma_inverse);

        } // End loop over particles
    } // End primordial chemistry if statement

    return 1;
}