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

int write_results_to_file(int new_file, char filename[30], int matrix_size, int num_iter,
                          int num_teams, int num_threads, int num_threads_per_team, double avg_time)
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

        
        fprintf(fptr, "Matrix Dimensions | Number of Timing Iterations | Number of Teams | Number of Threads | Number of Threads per Team | Average Time (s)\n");
        fprintf(fptr, "-------------------------------------------------------------------------------------------------------------------------------------\n");
    // Append results to pre-existing file
    } else {
        if ((fptr = fopen(filename, "a")) == NULL) return 0;
    }
    
    // Write results to the file.
    fprintf(fptr, "%-18d|%-29d|%-17d|%-19d|%-28d|%-17g\n", matrix_size, num_iter, num_teams, num_threads, num_threads_per_team, avg_time);
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

double run_test(int num_teams, int matrix_side_length)
{
    square_matrix A, B, C;
    initialise_matrix(&A, matrix_side_length);
    initialise_matrix(&B, matrix_side_length);
    initialise_matrix(&C, matrix_side_length);
    
    double start_time, elapsed_time;
    
    #pragma omp declare mapper(square_matrix m) map(m.side_length, m.values_length, m.values[:m.values_length])

    start_time = omp_get_wtime();

    #pragma omp target enter data map(to: A, B, C)

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
    
    #pragma omp target update from(C)

    #pragma omp target exit data map(release: A, B, C)

    elapsed_time = omp_get_wtime() - start_time;

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


    // >>> USER SETTINGS <<<
    int num_iter = 100;
    int matrix_side_length = 10;
    char *filename = "benchmark.txt";
    // >>>>>>>>>><<<<<<<<<<


    int num_threads, num_teams, num_threads_per_team, new_file, pass;
    double avg_time;
    int num_shading_units = 18432;
    // int max_teams = num_shading_units * 1.1; 
    int max_teams = 10;

    for (int num_teams_requested=1; num_teams_requested<max_teams; num_teams_requested++)
    {
        avg_time = 0;
        for (int iter=0; iter<num_iter; iter++)
        {
            avg_time += run_test(num_teams_requested, matrix_side_length) / num_iter;
        }

        // Write results to file
        get_gpu_parameters(num_teams_requested, &num_threads, &num_teams, &num_threads_per_team);
        new_file = (num_teams_requested == 1) ? 1 : 0;
        pass = write_results_to_file(new_file, filename, matrix_side_length, num_iter, num_teams, num_threads,
                              num_threads_per_team, avg_time);
        if (!pass)
        {
            fprintf(stdout, "Unable to write results to file! Exiting...\n");
            exit(0);
        }
        fprintf(stdout, "num_teams_requested = %d\n", num_teams_requested);
    }
    
    return 1;
}