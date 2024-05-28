#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "grackle_types.h"
#include "grackle_chemistry_data.h"
#include "phys_constants.h"
#include "index_helper.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

int calculate_pressure(chemistry_data *my_chemistry,
                       chemistry_data_storage *my_rates,
                       code_units *my_units,
                       grackle_field_data *my_fields,
                       gr_float *pressure)
{
    if (!my_chemistry->use_grackle) return 1;

    double tiny_number = 1e-20;
    const grackle_index_helper ind_helper = _build_index_helper(my_fields);
    int outer_ind, index;

    #ifdef _OPENMP
        #ifdef CPU
            #pragma omp parallel for schedule(runtime) private(outer_ind,index)
        #elif GPU
            enter_calculate_pressure(1);
            #pragma omp target teams parallel distribute for num_teams(GPU_NUM_TEAMS)
        #endif
    #endif

    for (outer_ind=0; outer_ind < ind_helper.outer_ind_size; outer_ind++)
    {
        const grackle_index_range range = _inner_range(outer_ind, &ind_helper);

        for (index = range.start; index <= range.end; index++)
        {
            pressure[index] = (my_chemistry->Gamma -1.0) * my_fields->density[index]
                                * my_fields->internal_energy[index];

            if (pressure[index] < tiny_number) pressure[index] = tiny_number;
        } // End loop over inner index
    } // End loop over outer index

    #if defined(_OPENMP) && defined(GPU)
        exit_calculate_pressure(1);
    #endif

    // Correct Gamma from H2
    if (my_chemistry->primordial_chemistry > 1)
    {
        // Calculate temperature units
        double temperature_units = get_temperature_units(my_units);

        double number_density, nH2, GammaH2Inverse, x, Gamma1, temp;
        double GammaInverse = 1.0 / (my_chemistry->Gamma - 1.0);

        #ifdef _OPENMP
            #ifdef CPU
                #pragma omp parallel for schedule(runtime) \
                private(outer_ind, index, number_density, nH2, GammaH2Inverse,\
                x, Gamma1, temp)
            #elif GPU
                enter_calculate_pressure(2);
                #pragma omp target teams distribute parallel for num_teams(GPU_NUM_TEAMS)
            #endif
        #endif

        for (outer_ind = 0; outer_ind < ind_helper.outer_ind_size; outer_ind++)
        {
            const grackle_index_range range = _inner_range(outer_ind, &ind_helper);

            for (index = range.start; index <= range.end; index++)
            {
                number_density = 0.25 * (my_fields->HeI_density[index]
                                         + my_fields->HeII_density[index]
                                         + my_fields->HeIII_density[index])
                                 + my_fields->HI_density[index]
                                 + my_fields->HII_density[index]
                                 + my_fields->HM_density[index]
                                 + my_fields->e_density[index];

                nH2 = 0.5 * (my_fields->H2I_density[index] + my_fields->H2II_density[index]);

                // First, approximate temperature
                if (number_density == 0) number_density = tiny_number;
                temp = max(temperature_units * pressure[index] / (number_density + nH2), 1);

                // Only do full computation if there is a reasonable amount of H2.
                // The second term in GammaH2Inverse accounts for the vibrational
                //  degrees of freedom.

                GammaH2Inverse = 0.5 * 5.0;
                if (nH2 / number_density > 1e-3)
                {
                    x = 6100.0 / temp;
                    if (x < 10.0)
                    {
                        GammaH2Inverse = 0.5 * (5 + 2.0 * POW(x, 2)
                                                 * exp(x)/POW(exp(x) - 1.0, 2));
                    }
                }

                Gamma1 = 1.0 + (nH2 + number_density)
                          / (nH2 * GammaH2Inverse + number_density * GammaInverse);

                // Correct pressure with improved Gamma.

                pressure[index] *= (Gamma1 - 1.0) / (my_chemistry->Gamma = 1.0);
            } // End loop over inner index
        } // End loop over outer index

        #if defined(_OPENMP) && defined(GPU)
            exit_calculate_pressure(2);
        #endif
    }
    return 1;
}


int main(int argc, char *argv[])
{
    

    return 1;
}