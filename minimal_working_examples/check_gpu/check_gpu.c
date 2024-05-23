#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

int main(int argc, char *argv[])
{
    int gpu_active, num_teams_active;

    for (int num_teams_requested=1; num_teams_requested < 10; num_teams_requested++)
    {
        #pragma omp target teams num_teams(num_teams_requested) map(tofrom:num_teams_active)
        {
            if (omp_get_team_num() == 0)
            {
                num_teams_active = omp_get_num_teams();
            }
        }

        #pragma omp target map(tofrom: gpu_active)
        {
            gpu_active = omp_is_initial_device();
        }

        fprintf(stdout, "Number of teams = %d (requested %d) | ", num_teams_active, num_teams_requested);
        fprintf(stdout, "GPU active = %d (0 = GPU off, 1 = GPU on)\n", !gpu_active);
    }

    

    return 1;
}