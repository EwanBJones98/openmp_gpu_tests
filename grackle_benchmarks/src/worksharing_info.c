#include <stdio.h>
#include <stdlib.h>

#include "worksharing_info.h"

#ifdef _OPENMP
    #include <omp.h>
#endif


void get_parallel_mode(char *parallel_mode)
{
    #if defined(GPU) && defined(_OPENMP)
        strcpy(parallel_mode, "GPU");
    #elif defined(CPU) && defined(_OPENMP)
        strcpy(parallel_mode, "CPU");
    #elif defined(SERIAL) && !defined(_OPENMP)
        strcpy(parallel_mode, "serial");
    #else
        fprintf(stderr, "Unknown parallel mode active. Exiting...\n");
        exit(0);
    #endif
}


void gpu_default_worksharing_info(int *gpu_active, int *num_teams, int *num_threads_per_team)
{
    *gpu_active = omp_get_default_device();

    #pragma omp target teams map(tofrom:num_teams, num_threads_per_team)
    {
        *num_teams = omp_get_num_teams();
        #pragma omp parallel
        {
            if (omp_get_team_num() == 0)
                *num_threads_per_team = omp_get_num_threads();
        }
    }
}


int check_gpu_worksharing(int *nTeams, int *nThreadsPerTeam)
{
    int result = 1;

    int _nTeams, _nThreadsPerTeam;
    #pragma omp target teams map(tofrom:_nTeams, _nThreadsPerTeam) \
        num_teams(*nTeams)
    {
        _nTeams = omp_get_num_teams();
        #pragma omp parallel num_threads(*nThreadsPerTeam)
        #pragma omp single
            _nThreadsPerTeam = omp_get_num_threads();
    }

    if (_nTeams != *nTeams)
    {
        fprintf(stderr, "Number of teams requested: %d\n", *nTeams);
        fprintf(stderr, "Number of teams allocated: %d\n", _nTeams);
        result = 0;
    }

    if (_nThreadsPerTeam != *nThreadsPerTeam)
    {
        fprintf(stderr, "Number of threads per team requested: %d\n", *nThreadsPerTeam);
        fprintf(stderr, "Number of threads per team allocated: %d\n", _nThreadsPerTeam);
        result = 0;
    }

    return result;
}