#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#ifdef _OPENMP
    #include <omp.h>
#endif


typedef struct
{
    double *values;
    int values_length;
    int side_length;

} square_matrix;



int* logspace_int(int start, int end, int num_values)
{
    int *result;

    result = malloc(sizeof(int) * num_values);
    double spacing = (log(end) - log(start)) / (num_values-1);

    for (int i=0; i<num_values; i++)
    {
        result[i] = (int) exp(log(start) + spacing * i);
    }
    
    return result;
}

int* linspace_int(int start, int end, int num_values)
{
    int *result;
    result = malloc(sizeof(int) * num_values);

    double spacing = (end - start) / (num_values-1);

    for (int i=0; i<num_values; i++)
    {
        result[i] = (int) (start + spacing * i);
    }

    return result;
}


void get_gpu_parameters(int num_teams_requested, int *num_threads, int *num_teams, int *num_threads_per_team)
{
    int n_threads, n_teams, n_threads_per_team;
    #pragma omp target teams map(tofrom:n_threads,n_teams,n_threads_per_team) num_teams(num_teams_requested)
    {
        #pragma omp parallel
        {
            if (omp_get_team_num() == 0)
            {
                n_threads_per_team = omp_get_num_threads();
                n_teams            = omp_get_num_teams();
                n_threads          = n_threads_per_team * n_teams;
            }
            
        }
    }
        
    *num_threads          = n_threads;
    *num_teams            = n_teams;
    *num_threads_per_team = n_threads_per_team;
}

int check_gpu_memory(int matrix_side_length, double vram_gb)
{
    size_t max_matrix_size = sizeof(double) * matrix_side_length * matrix_side_length + 2 * sizeof(int);
    size_t vram_avail      = vram_gb * 1e9; 
    if (3*max_matrix_size >= vram_avail)
    {
        int largest_possible_matrix_dim = (int) sqrt((vram_avail/3 - 2 * sizeof(int)) / sizeof(double));
        fprintf(stdout, "Matrix of dimension %dx%d is too large!\n", matrix_side_length, matrix_side_length); 
        fprintf(stdout, "Largest theoretical dimension of matrix supported with %f GB of vram is %dx%d!\n", vram_gb,
                        largest_possible_matrix_dim, largest_possible_matrix_dim);
        return 0;
    }
    return 1;
}

int write_results_to_file(int new_file, char filename[30], int matrix_size, int num_iter,
                          int num_teams, int num_threads, int num_threads_per_team,
                          double total_time, double transfer_time, double calculation_time)
{
    /*
        new_file = 1 --> create new file
        new_file = 0 --> append to existing file
        
    */

    FILE *fptr;
    
    // Create new file and write table header
    if (new_file)
    {
        if ((fptr = fopen(filename, "w+")) == NULL) return 0;

        
        fprintf(fptr, "Matrix Dimensions | Number of Timing Iterations | Number of Teams | Number of Threads | Number of Threads per Team | Total Time (s) | Data Transfer Time (s) | Calculation Time (s)\n");
        fprintf(fptr, "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Append results to pre-existing file
    } else {
        if ((fptr = fopen(filename, "a")) == NULL) return 0;
    }
    
    // Write results to the file.
    fprintf(fptr, "%-18d|%-29d|%-17d|%-19d|%-28d|%-17g|%-24g|%-21g\n", matrix_size, num_iter, num_teams, num_threads, num_threads_per_team, total_time, transfer_time, calculation_time);
    fclose(fptr);

    return 1;
}

void initialise_matrix(square_matrix *M, int side_length)
{
    M->side_length = side_length;
    M->values_length = side_length * side_length;
    
    M->values = malloc(sizeof(double) * M->values_length);
    for (int i=0; i<M->values_length; i++)
    {
        M->values[i] = i;
    }
}

int is_close(double x, double y, double tolerance)
{
    return fabs(x - y) <= tolerance;
}

int matrix_is_close(square_matrix *result, square_matrix *expected_result)
{
    if  (!is_close(result->side_length, expected_result->side_length, 1e-10)) return 0;

    for (int i=0; i<result->values_length; i++)
    {
        if (!is_close(result->values[i], expected_result->values[i], 1e-10)) return 0;
    }

    return 1;
}

void map_to(square_matrix *A, square_matrix *B, square_matrix *C)
{
    #pragma omp target enter data map(alloc: A->side_length, A->values[:A->values_length],\
                                             B->values[:B->values_length],\
                                             C->side_length, C->values[:C->values_length])

    #pragma omp target update to(A->side_length, A->values[:A->values_length],\
                                 B->values[:B->values_length],\
                                 C->side_length, C->values[:C->values_length])
}

void map_from(square_matrix *A, square_matrix *B, square_matrix *C)
{
    #pragma omp target update from(C->values[:C->values_length])

    #pragma omp target exit data map(delete: A->side_length, A->values[:A->values_length],\
                                             B->values[:B->values_length],\
                                             C->side_length, C->values[:C->values_length])
}

double run_test(int num_teams, int matrix_side_length, double *transfer_time, double *calculation_time)
{
    square_matrix A, B, C;
    initialise_matrix(&A, matrix_side_length);
    initialise_matrix(&B, matrix_side_length);
    initialise_matrix(&C, matrix_side_length);
    
    double start_time, elapsed_time, start_time_calc;

    start_time = omp_get_wtime();

    map_to(&A, &B, &C);

    start_time_calc = omp_get_wtime();

    #pragma omp target teams distribute parallel for collapse(2) num_teams(num_teams)
    for (int i=0; i<C.side_length; i++)
    {
        for (int j=0; j<C.side_length; j++)
        {
            for (int k=0; k<A.side_length; k++)
            {   
                if (k == 0) C.values[i*C.side_length+j] = 0;
                C.values[i*C.side_length + j] += A.values[i*A.side_length + k] * B.values[k*B.side_length + j];
            }
        }
    }

    *calculation_time = omp_get_wtime() - start_time_calc;
    
    map_from(&A, &B, &C);

    elapsed_time = omp_get_wtime() - start_time;

    *transfer_time = elapsed_time - *calculation_time;

    // Check that C has been updated from its initial values
    if (matrix_is_close(&C, &A))
    {
        fprintf(stdout, "Matrix not updated on GPU. Exiting...\n");
        exit(0);
    }

    // Check that number of teams can be set to this value
    int actual_num_teams;
    actual_num_teams = -1;
    #pragma omp target teams map(tofrom:actual_num_teams) num_teams(num_teams)
    {
        if (omp_get_team_num() == 0) actual_num_teams = omp_get_num_teams();
    }

    if (num_teams != actual_num_teams)
    {
        fprintf(stdout, "requested teams = %d, actual teams = %d\n", num_teams, actual_num_teams);
    }

    // Free memory
    free(A.values);
    free(B.values);
    free(C.values);

    return elapsed_time;
}

int main(int argc, char *argv[])
{
    // GPU specs (6000 ada lovelace)
    int num_shading_units = 18432;
    double vram_gb        = 48;

    // >>> USER SETTINGS <<<
    int num_iter = 10;
    char *filename = "benchmark.txt";
    
    int min_teams          = 500;
    int max_teams          = 2 * num_shading_units;
    int num_team_intervals = 5;
    int log_team_spacing   = 1;

    int min_dims           = 10;
    int max_dims           = 10000; //44000;
    int num_dim_intervals  = 10;
    int log_dim_spacing    = 1;
    // >>>>>>>>>><<<<<<<<<<


    if (!check_gpu_memory(max_dims, vram_gb)) exit(0);

    int *dims      = (log_dim_spacing==1) ? logspace_int(min_dims, max_dims, num_dim_intervals) : linspace_int(min_dims, max_dims, num_dim_intervals);
    int *num_teams = (log_team_spacing==1) ? logspace_int(min_teams, max_teams, num_team_intervals) : linspace_int(min_teams, max_teams, num_dim_intervals);
    
    int new_file, pass, num_teams_actual, num_threads_actual, num_threads_per_team_actual;
    int num_teams_requested=-1, matrix_dims_requested=-1;
    double total_time, transfer_time, calculation_time;
    double transfer_time_buffer, calculation_time_buffer;

    for (int i=0; i<num_team_intervals; i++)
    {
        if (num_teams_requested == num_teams[i]) continue;
        num_teams_requested = num_teams[i];

        for (int j=0; j<num_dim_intervals; j++)
        {
            total_time = 0;
            transfer_time = 0;
            calculation_time = 0;

            if (matrix_dims_requested == dims[j]) continue;
            matrix_dims_requested = dims[j];

            for (int iter=0; iter<num_iter; iter++)
            {
                total_time += run_test(num_teams_requested, matrix_dims_requested, &transfer_time_buffer, &calculation_time_buffer) / num_iter;

                transfer_time    += transfer_time_buffer / num_iter;
                calculation_time += calculation_time_buffer / num_iter;
            }

            fprintf(stdout, "Finished test for num_teams=%d, matrix_dim=%d\n", num_teams_requested, matrix_dims_requested);

            // Write results to file
            get_gpu_parameters(num_teams_requested, &num_threads_actual, &num_teams_actual, &num_threads_per_team_actual);
            new_file = (i == 0 && j==0) ? 1 : 0;
            pass = write_results_to_file(new_file, filename, matrix_dims_requested, num_iter, num_teams_actual, num_threads_actual,
                                num_threads_per_team_actual, total_time, transfer_time, calculation_time);
            if (!pass)
            {
                fprintf(stdout, "Unable to write results to file! Exiting...\n");
                exit(0);
            }
        }
    }

    free(dims);
    free(num_teams);

    return 1;
}