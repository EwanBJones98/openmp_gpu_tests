#ifndef CALCULATE_PRESSURE_H
#define CALCULATE_PRESSURE_H

void enter_calculate_pressure(gr_float *pressure,
                              double temperature_units,
                              double GammaInverse,
                              double tiny_number,
                              chemistry_data *my_chemistry,
                              grackle_field_data *my_fields,
                              code_units *my_units,
                              const grackle_index_helper *ind_helper);

void exit_calculate_pressure(gr_float *pressure,
                             double temperature_units,
                             double GammaInverse,
                             double tiny_number,
                             chemistry_data *my_chemistry,
                             grackle_field_data *my_fields,
                             code_units *my_units,
                             const grackle_index_helper *ind_helper);

#ifdef _OPENMP

    void omp_cpu_info(int *num_threads);

    void omp_gpu_info(int *gpu_active, int *num_threads, int *num_teams);
    
#endif

#endif