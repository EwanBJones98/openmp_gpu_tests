#include "grackle_chemistry_data.h"
#include "grackle_types.h"
#include "index_helper.h"
#include "calculate_pressure_gpu.h"

#ifdef _OPENMP
    #include <omp.h>
#endif


void enter_calculate_pressure(gr_float *pressure,
                              double temperature_units,
                              double GammaInverse,
                              double tiny_number,
                              chemistry_data *my_chemistry,
                              grackle_field_data *my_fields,
                              code_units *my_units,
                              const grackle_index_helper *ind_helper)
{
    int length = 1;
    for (int i=0; i<my_fields->grid_rank; i++)
    {
        length *= my_fields->grid_dimension[i];
    }

    if (my_chemistry->primordial_chemistry <= 1)
    {
        #pragma omp target enter data\
                map(alloc:pressure[:length],\
                          tiny_number,\
                          my_chemistry->Gamma,\
                          my_chemistry->primordial_chemistry,\
                          my_fields->density[:length],\
                          my_fields->internal_energy[:length],\
                          ind_helper->i_start,\
                          ind_helper->j_start,\
                          ind_helper->k_start,\
                          ind_helper->i_end,\
                          ind_helper->i_dim,\
                          ind_helper->j_dim,\
                          ind_helper->num_j_inds)

        #pragma omp target update\
                to(tiny_number,\
                   my_chemistry->Gamma,\
                   my_chemistry->primordial_chemistry,\
                   my_fields->density[:length],\
                   my_fields->internal_energy[:length],\
                   ind_helper->i_start,\
                   ind_helper->j_start,\
                   ind_helper->k_start,\
                   ind_helper->i_end,\
                   ind_helper->i_dim,\
                   ind_helper->j_dim,\
                   ind_helper->num_j_inds)
    } else {

        #pragma omp target enter data\
                map(alloc:pressure[:length],\
                          tiny_number,\
                          temperature_units,\
                          GammaInverse,\
                          my_chemistry->Gamma,\
                          my_chemistry->primordial_chemistry,\
                          my_fields->density[:length],\
                          my_fields->internal_energy[:length],\
                          ind_helper->i_start,\
                          ind_helper->j_start,\
                          ind_helper->k_start,\
                          ind_helper->i_end,\
                          ind_helper->i_dim,\
                          ind_helper->j_dim,\
                          ind_helper->num_j_inds,\
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
                to(temperature_units,\
                   GammaInverse,\
                   tiny_number,\
                   my_chemistry->Gamma,\
                   my_chemistry->primordial_chemistry,\
                   my_fields->density[:length],\
                   my_fields->internal_energy[:length],\
                   ind_helper->i_start,\
                   ind_helper->j_start,\
                   ind_helper->k_start,\
                   ind_helper->i_end,\
                   ind_helper->i_dim,\
                   ind_helper->j_dim,\
                   ind_helper->num_j_inds,\
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
}


void exit_calculate_pressure(gr_float *pressure,
                             double temperature_units,
                             double GammaInverse,
                             double tiny_number,
                             chemistry_data *my_chemistry,
                             grackle_field_data *my_fields,
                             code_units *my_units,
                             const grackle_index_helper *ind_helper)
{
    int length = 1;
    for (int i=0; i<my_fields->grid_rank; i++)
    {
        length *= my_fields->grid_dimension[i];
    }

    #pragma omp target update from(pressure[:length])


    if (my_chemistry->primordial_chemistry <= 1)
    {
        #pragma omp target exit data map(delete:pressure[:length],\
                                                tiny_number,\
                                                my_chemistry->Gamma,\
                                                my_chemistry->primordial_chemistry,\
                                                my_fields->density[:length],\
                                                my_fields->internal_energy[:length],\
                                                ind_helper->i_start,\
                                                ind_helper->j_start,\
                                                ind_helper->k_start,\
                                                ind_helper->i_end,\
                                                ind_helper->i_dim,\
                                                ind_helper->j_dim,\
                                                ind_helper->num_j_inds)

    } else {

        #pragma omp target exit data map(delete:pressure[:length],\
                                                tiny_number,\
                                                temperature_units,\
                                                GammaInverse,\
                                                my_chemistry->Gamma,\
                                                my_chemistry->primordial_chemistry,\
                                                my_fields->density[:length],\
                                                my_fields->internal_energy[:length],\
                                                ind_helper->i_start,\
                                                ind_helper->j_start,\
                                                ind_helper->k_start,\
                                                ind_helper->i_end,\
                                                ind_helper->i_dim,\
                                                ind_helper->j_dim,\
                                                ind_helper->num_j_inds,\
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
}

#ifdef _OPENMP
    void omp_cpu_info(int *num_threads)
    {   
        int n=0;
        #pragma omp parallel reduction(+:n)
        n += 1;
        *num_threads = n;
    }

    void omp_gpu_info(int *gpu_active, int *num_threads, int *num_teams)
    {
        int num_threads_per_team = 0;
        int n_teams=0;
        #pragma omp target teams map(tofrom:num_threads_per_team, n_teams)
        {
            n_teams = omp_get_num_teams();
            #pragma omp parallel
            {
                if (omp_get_team_num() == 0)
                    num_threads_per_team = omp_get_num_threads();
            }
        }
        *num_teams = n_teams;
        *num_threads = num_threads_per_team * *num_teams;
        *gpu_active = omp_get_default_device();
    }
#endif