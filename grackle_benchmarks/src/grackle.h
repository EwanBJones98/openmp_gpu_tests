/***********************************************************************
/
/ Grackle function prototypes
/
/
/ Copyright (c) 2013, Enzo/Grackle Development Team.
/
/ Distributed under the terms of the Enzo Public Licence.
/
/ The full license is in the file LICENSE, distributed with this 
/ software.
************************************************************************/

#ifndef __GRACKLE_H__
#define __GRACKLE_H__

#include "grackle_types.h"
#include "grackle_chemistry_data.h"


void set_velocity_units(code_units *my_units);

double get_temperature_units(code_units *my_units);

int set_default_chemistry_parameters(chemistry_data *my_grackle);

int local_initialize_chemistry_data(chemistry_data *my_chemistry, 
                                    chemistry_data_storage *my_rates,
                                    code_units *my_units);

int local_free_chemistry_data(chemistry_data *my_chemistry,
                              chemistry_data_storage *my_rates);

#endif