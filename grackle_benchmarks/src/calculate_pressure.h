#ifndef _CALCULATE_PRESSURE_H
#define _CALCULATE_PRESSURE_H

void calculate_pressure(double *pressure, int field_length,
                        chemistry_data *my_chemistry,
                        chemistry_data_storage *my_rates,
                        grackle_field_data *my_fields,
                        code_units *my_units,
                        int nTeams, int nThreadsPerTeam);

void enter_calculate_pressure(double *pressure,
                              chemistry_data *my_chemistry,
                              grackle_field_data *my_fields,
                              code_units *my_units);

void exit_calculate_pressure(double *pressure,
                             chemistry_data *my_chemistry,
                             grackle_field_data *my_fields,
                             code_units *my_units);

#endif