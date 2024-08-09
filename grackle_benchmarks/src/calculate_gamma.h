#ifndef _CALCULATE_GAMMA_H
#define _CALCULATE_GAMMA_H

int calculate_gamma(double *gamma,
                    int field_length,
                    chemistry_data *my_chemistry,
                    chemistry_data_storage *my_rates,
                    grackle_field_data *my_fields,
                    code_units *my_units,
                    int nTeams,
                    int nThreadsPerTeam);

void enter_calculate_gamma(double *gamma,
                           chemistry_data *my_chemistry,
                           grackle_field_data *my_fields,
                           code_units *my_units);

void exit_calculate_gamma(double *gamma,
                          chemistry_data *my_chemistry,
                          grackle_field_data *my_fields,
                          code_units *my_units);

#endif