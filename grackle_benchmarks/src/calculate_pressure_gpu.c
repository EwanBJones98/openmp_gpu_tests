#include <stdio.h>
#include <stdlib.h>
#include "grackle_chemistry_data.h"
#include "grackle_types.h"
#include "calculate_pressure.h"

#ifdef _OPENMP
    #include <omp.h>
#endif


void enter_calculate_pressure(double *pressure,
                              chemistry_data *my_chemistry,
                              grackle_field_data *my_fields,
                              code_units *my_units)
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
                          my_chemistry->primordial_chemistry,\
                          my_chemistry->Gamma,\
                          my_fields->density[:length],\
                          my_fields->internal_energy[:length])

        #pragma omp target update\
                to(my_chemistry->primordial_chemistry,\
                   my_chemistry->Gamma,\
                   my_fields->density[:length],\
                   my_fields->internal_energy[:length])
    } else {

        #pragma omp target enter data\
                map(alloc:pressure[:length],\
                          my_chemistry->Gamma,\
                          my_chemistry->primordial_chemistry,\
                          my_fields->density[:length],\
                          my_fields->internal_energy[:length],\
                          my_fields->HeI_density[:length],\
                          my_fields->HeII_density[:length],\
                          my_fields->HeIII_density[:length],\
                          my_fields->HI_density[:length],\
                          my_fields->HII_density[:length],\
                          my_fields->HM_density[:length],\
                          my_fields->e_density[:length],\
                          my_fields->H2I_density[:length],\
                          my_fields->H2II_density[:length],\
                          my_units->temperature_units)

        #pragma omp target update\
                to(my_chemistry->Gamma,\
                   my_chemistry->primordial_chemistry,\
                   my_fields->density[:length],\
                   my_fields->internal_energy[:length],\
                   my_fields->HeI_density[:length],\
                   my_fields->HeII_density[:length],\
                   my_fields->HeIII_density[:length],\
                   my_fields->HI_density[:length],\
                   my_fields->HII_density[:length],\
                   my_fields->HM_density[:length],\
                   my_fields->e_density[:length],\
                   my_fields->H2I_density[:length],\
                   my_fields->H2II_density[:length],\
                   my_units->temperature_units)
    }
}


void exit_calculate_pressure(double *pressure,
                             chemistry_data *my_chemistry,
                             grackle_field_data *my_fields,
                             code_units *my_units)
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
                                            my_chemistry->Gamma,\
                                            my_chemistry->primordial_chemistry,\
                                            my_fields->density[:length],\
                                            my_fields->internal_energy[:length])

    } else {

        #pragma omp target exit data map(delete:pressure[:length],\
                                            my_chemistry->Gamma,\
                                            my_chemistry->primordial_chemistry,\
                                            my_fields->density[:length],\
                                            my_fields->internal_energy[:length],\
                                            my_fields->HeI_density[:length],\
                                            my_fields->HeII_density[:length],\
                                            my_fields->HeIII_density[:length],\
                                            my_fields->HI_density[:length],\
                                            my_fields->HII_density[:length],\
                                            my_fields->HM_density[:length],\
                                            my_fields->e_density[:length],\
                                            my_fields->H2I_density[:length],\
                                            my_fields->H2II_density[:length],\
                                            my_units->temperature_units)
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

    void omp_gpu_info(int *gpu_active, int *num_threads, int *num_teams, int update_vals)
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
        if (update_vals == 1)
        {
            *num_teams = n_teams;
            *num_threads = num_threads_per_team * *num_teams;
        }
        
        *gpu_active = omp_get_default_device();
    }

    int omp_gpu_check_worksharing_config(int *num_teams_requested, int *num_threads_per_team_requested)
    {
        int pass=1;
        
        int num_teams_actual, num_threads_per_team_actual;
        #pragma omp target teams map(tofrom:num_teams_actual, num_threads_per_team_actual) \
            num_teams(*num_teams_requested)
        {
            num_teams_actual = omp_get_num_teams();
            #pragma omp parallel num_threads(*num_threads_per_team_requested)
            #pragma omp single
                num_threads_per_team_actual = omp_get_num_threads();
        }

        if ((*num_teams_requested != num_teams_actual) || (*num_threads_per_team_requested != num_threads_per_team_actual))
        {
            fprintf(stdout, "\n** WARNING: skipping worksharing configuration **\n");
            fprintf(stdout, "\tnum teams requested = %d -- num teams initialised = %d\n", *num_teams_requested, num_teams_actual);
            fprintf(stdout, "\tnum threads per team requested = %d -- num threads per team initialised = %d\n",
                        *num_threads_per_team_requested, num_threads_per_team_actual);
            pass = 0;
        } else if (omp_get_default_device() != 0) {
            fprintf(stdout, "GPU inactive! Exiting...\n");
            exit(0);
        }

        return pass;
    }
#endif