#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

void check_grid(double *grid, int grid_size, double expected_value, double tolerance)
{
    for (int i=0; i<grid_size; i++)
        if (grid[i] - expected_value > tolerance)
        {
            fprintf(stderr, "Grid calculation failed!\n\tExpected result=%g\n\tCalculated result=%g\nExiting...\n",
                        expected_value, grid[i]);
            exit(0);
        }

}

int get_grid_size(int grid_rank, int *grid_dims)
{
    int grid_size = 1;
    for (int rank=0; rank<grid_rank; rank++)
        grid_size *= grid_dims[rank];

    return grid_size;
}

void set_grid(double *grid, int grid_size, double value)
{
    for (int i=0; i<grid_size; i++)
        grid[i] = value;
}

void reset_for_next_test(double *result_grid, int grid_size, double expected_value)
{
    // Check result grid is calculated to expected value
    #pragma omp target update from(result_grid[:grid_size])
    check_grid(result_grid, grid_size, expected_value, 1e-10);

    // Refresh result grid for next test
    set_grid(result_grid, grid_size, -1);
    #pragma omp target update to(result_grid[:grid_size])
}


int main(int argc, char *argv[])
{
    // Hardware properties
    int num_multiprocessors = 142;
    int max_threads_per_multiprocessor = 1536;
    int max_threads_per_block = 1024;

    // OpenMP options
    int num_teams_per_multiprocessor = 3;
    int num_threads_per_team = max_threads_per_multiprocessor / num_teams_per_multiprocessor;
    int num_teams            = num_teams_per_multiprocessor * num_multiprocessors;
    if (num_threads_per_team > max_threads_per_block)
    {
        fprintf(stderr, "Requested %d threads per team, but hardware maximum is %d\nExiting...\n",
                    num_threads_per_team, max_threads_per_block);
        exit(0);
    }

    // Grid options
    int grid_rank = 3;
    int grid_dims[] = {100,100,100};
    int grid_size = get_grid_size(grid_rank, grid_dims);

    // Create three grids named x,y,z
    double *x, *y, *z;
    x = malloc(sizeof(double) * grid_size);
    y = malloc(sizeof(double) * grid_size);
    z = malloc(sizeof(double) * grid_size);
    
    // Set grid values
    double x_value=44.44, y_value=16.16;
    double z_value_expected = x_value * y_value;
    set_grid(x, grid_size, x_value);
    set_grid(y, grid_size, y_value);
    set_grid(z, grid_size, -1.);

    // Map data from cpu to gpu
    #pragma omp target enter data map(alloc:x[:grid_size], y[:grid_size], z[:grid_size])
    #pragma omp target update to(x[:grid_size], y[:grid_size], z[:grid_size])

    // Run tests multiple times to calculate an average time
    int num_iter = 1000;
    double timings_default[num_iter], timings_custom[num_iter];

    for (int iter=0; iter<num_iter; iter++)
    {

        // Calculate with default omp behaviour
        double start_time, end_time;
        start_time = omp_get_wtime();
        #pragma omp target teams distribute parallel for
        for (int i=0; i<grid_size; i++)
            z[i] = x[i] * y[i];
        end_time = omp_get_wtime();
        timings_default[iter] = end_time - start_time;

        reset_for_next_test(z, grid_size, z_value_expected);

        // Calculate with custom omp behaviour set by user
        start_time = omp_get_wtime();
        #pragma omp target teams distribute parallel for num_teams(num_teams) num_threads(num_threads_per_team)
        for (int i=0; i<grid_size; i++)
            z[i] = x[i] * y[i];
        end_time = omp_get_wtime();
        timings_custom[iter] = end_time - start_time;

        reset_for_next_test(z, grid_size, z_value_expected);

        // Calculate with grid stride loop approach
        //! >> Not yet implemented <<
    }

    // Calculate average times for each looping method
    double mean_default=0, mean_custom=0;
    for (int iter=0; iter<num_iter; iter++)
    {
        mean_default += timings_default[iter] / num_iter;
        mean_custom  += timings_custom[iter] / num_iter;
    }
    
    double stdev_default=0, stdev_custom=0;
    for (int iter=0; iter<num_iter; iter++)
    {
        stdev_default += pow(timings_default[iter] - mean_default, 2);
        stdev_custom  += pow(timings_custom[iter] - mean_custom, 2);
    }
    stdev_default = sqrt(stdev_default/num_iter);
    stdev_custom  = sqrt(stdev_custom/num_iter);


    fprintf(stdout, "Time taken (default) = %g ± %g \nTime taken (custom) = %g ± %g\n",
                    mean_default, stdev_default, mean_custom, stdev_custom);

    return 0;
}