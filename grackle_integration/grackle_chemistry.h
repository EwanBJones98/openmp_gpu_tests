#include <stdbool.h>

#ifndef _GRACKLE_CHEMISTRY
#define _GRACKLE_CHEMISTRY

typedef struct
{
    bool use_grackle;

    int primordial_chemistry;

    int dust_chemistry;
    int h2_on_dust;

    double temperature_start;
    double temperature_end;
    int number_of_temperature_bins;

    double local_dust_to_gas_ratio;

    bool use_isrf_field;
    double interstellar_radiation_field;
} grackle_chemistry;

#endif 