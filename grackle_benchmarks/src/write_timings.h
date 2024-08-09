#ifndef _WRITE_TIMINGS_H
#define _WRITE_TIMINGS_H

int write_timings(int num_timing_iter, double *calc_times, double *data_times, int num_teams,
                   int num_threads_per_team, char mode[10], grackle_field_data *my_fields,
                   chemistry_data *my_chemistry, int new_file, char *filename);

#endif