#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    #elif !defined(_OPENMP)
        strcpy(parallel_mode, "serial");
    #else
        fprintf(stderr, "Unknown parallel mode active. Exiting...\n");
        exit(0);
    #endif
}

#ifdef GPU
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
        int pass = 1;

        int req_nTeams          = *nTeams;
        int req_nThreadsPerTeam = *nThreadsPerTeam;

        int _nTeams, _nThreadsPerTeam;
        #pragma omp target teams distribute parallel for map(to:req_nTeams, req_nThreadsPerTeam) map(tofrom:_nTeams, _nThreadsPerTeam)\
            num_teams(req_nTeams) num_threads(req_nThreadsPerTeam)
        for (int i=0; i<10000; i++)
        {
            if(pass == 0) continue;

            _nTeams          = omp_get_num_teams();
            _nThreadsPerTeam = omp_get_num_threads();

            if (_nThreadsPerTeam != req_nThreadsPerTeam) pass = 0;
            if (_nTeams != req_nTeams) pass = 0;
        }

        fprintf(stderr, "Number of teams requested/allocated: %d/%d\n", req_nTeams, _nTeams);
        fprintf(stderr, "Number of threads per team requested/allocated: %d/%d\n", req_nThreadsPerTeam, _nThreadsPerTeam);

        *nTeams = _nTeams;
        *nThreadsPerTeam = _nThreadsPerTeam;

        return pass;
    }
#endif

#ifdef CPU
    int check_cpu_worksharing(int *num_teams)
    {
        // + num_teams here = number of CPU cores
        // + each CPU core runs one thread

        int n=0;
        #pragma omp parallel reduction(+:n)
        n += 1;

        if (*num_teams != n)
        {
            fprintf(stderr, "Number of CPU cores requested: %d\n", *num_teams);
            fprintf(stderr, "Number of CPU cores allocated: %d\n", n);
            return 0;
        }
        return 1;
    }
#endif