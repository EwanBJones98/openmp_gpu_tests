#include <stdbool.h>

#ifndef _GRACKLE_UNITS
#define _GRACKLE_UNITS

typedef struct
{
    double time_units;
    double length_units;
    double density_units;
    double velocity_units;
    double temperature_units;

    double co_length_units;
    double co_density_units;

    bool comoving_coordinates;
    double a_units;
    double a_value;
    
} grackle_units;

// Function prototypes from grackle_units.c
void set_comoving_units(grackle_units *my_units);
void set_velocity_units(grackle_units *my_units);
void set_temperature_units(grackle_units *my_units);

#endif