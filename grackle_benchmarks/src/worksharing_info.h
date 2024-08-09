#ifndef _WORKSHARING_INFO_H
#define _WORKSHARING_INFO_H

void get_parallel_mode(char *parallel_mode);

void gpu_default_worksharing_info(int *gpu_active, int *num_teams, int *num_threads_per_team);

void check_gpu_worksharing(int *nTeams, int *nThreadsPerTeam);

#endif