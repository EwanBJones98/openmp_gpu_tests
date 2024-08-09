#ifndef _CALCULATE_GAMMA_GPU_H
#define _CALCULATE_GAMMA_GPU_H


void enter_calculate_gamma(gr_float *gamma,
                            double *gamma_inverse,
                            chemistry_data *my_chemistry,
                            grackle_field_data *my_fields,
                            code_units *my_units);

void exit_calculate_gamma(gr_float *gamma,
                            double *gamma_inverse,
                            chemistry_data *my_chemistry,
                            grackle_field_data *my_fields,
                            code_units *my_units);

void gpu_default_worksharing_info(int *gpu_active, int *num_teams,
                                    int *num_threads_per_team);

void check_gpu_worksharing(int *nTeams, int *nThreadsPerTeam);

#endif