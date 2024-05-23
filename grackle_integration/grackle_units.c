#include "grackle_units.h"
#include "physical_constants.h"
#include <stdbool.h>
#include <math.h>

void set_comoving_units(grackle_units *my_units)
{
    if (my_units->comoving_coordinates)
    {
        my_units->co_length_units  = my_units->length_units;
        my_units->co_density_units = my_units->density_units;
    } else {
        my_units->co_length_units  = my_units->length_units * 
                                    my_units->a_value * my_units->a_units;
        my_units->co_density_units = my_units->density_units /
                                    pow(my_units->a_units * my_units->a_value, 3);
    }
}

double get_velocity_units(grackle_units *my_units)
{
    double velocity_units;
    velocity_units = my_units->length_units / my_units->time_units;
    if (my_units->comoving_coordinates)
    {
        velocity_units /= my_units->a_value;
    }
    return velocity_units;
}

void set_velocity_units(grackle_units *my_units)
{
    my_units->velocity_units = get_velocity_units(my_units);
}

double get_temperature_units(grackle_units *my_units)
{
    double velocity_units;
    velocity_units = get_velocity_units(my_units);
    return mh * pow(velocity_units, 2) / kboltz;
}

void set_temperature_units(grackle_units *my_units)
{
    my_units->temperature_units = get_temperature_units(my_units);
}