#include <stdio.h>
#include <stdlib.h>
#include "grackle_chemistry_data.h"
#include "grackle_types.h"
#include "calculate_temperature.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

void enter_calculate_temperature(double *temperature,
                                 chemistry_data *my_chemistry,
                                 grackle_field_data *my_fields,
                                 code_units *my_units)
{
    int length = 1;
    for (int i=0; i<my_fields->grid_rank; i++)
    {
        length *= my_fields->grid_dimension[i];
    }

    #pragma omp target enter data \
        map(alloc:temperature[:length],\
            my_chemistry->primordial_chemistry,\
            my_fields->HeI_density[:length],\
            my_fields->HeII_density[:length],\
            my_fields->HeIII_density[:length],\
            my_fields->HI_density[:length],\
            my_fields->HII_density[:length],\
            my_fields->HM_density[:length],\
            my_fields->e_density[:length],\
            my_fields->H2I_density[:length],\
            my_fields->H2II_density[:length])

    #pragma omp target update\
        to(temperature[:length],\
            my_chemistry->primordial_chemistry,\
            my_fields->HeI_density[:length],\
            my_fields->HeII_density[:length],\
            my_fields->HeIII_density[:length],\
            my_fields->HI_density[:length],\
            my_fields->HII_density[:length],\
            my_fields->HM_density[:length],\
            my_fields->e_density[:length],\
            my_fields->H2I_density[:length],\
            my_fields->H2II_density[:length])
}

void exit_calculate_temperature(double *temperature,
                          chemistry_data *my_chemistry,
                          grackle_field_data *my_fields,
                          code_units *my_units)
{
    int length = 1;
    for (int i=0; i<my_fields->grid_rank; i++)
    {
        length *= my_fields->grid_dimension[i];
    }

    #pragma omp target update from(temperature[:length])

    #pragma omp target exit data \
        map(delete:temperature[:length],\
            my_chemistry->primordial_chemistry,\
            my_fields->HeI_density[:length],\
            my_fields->HeII_density[:length],\
            my_fields->HeIII_density[:length],\
            my_fields->HI_density[:length],\
            my_fields->HII_density[:length],\
            my_fields->HM_density[:length],\
            my_fields->e_density[:length],\
            my_fields->H2I_density[:length],\
            my_fields->H2II_density[:length])
}