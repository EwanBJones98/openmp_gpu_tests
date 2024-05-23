#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

int all_equal(double *array, int array_length, double expected_value, double tolerance)
{
    for (int i=0; i<array_length; i++)
    {
        if (fabs(array[i] - expected_value) > tolerance)
        {
            return 0;
        }
    }
    return 1;
}

void set_all_equal(double *array, int array_length, double value)
{
    for (int i=0; i<array_length; i++)
    {
        array[i] = value;
    }
}

int time_serial(double *array, int array_length, int number_of_iterations, 
                double *calc_time)
{
    for (int iter=0; iter<array_length; iter++)
    {
        double start_time, end_time;

        set_all_equal(array, array_length, 4);

        start_time = omp_get_wtime();

        for (int i=0; i<array_length; i++)
        {
            array[i] = -1;
        }

        end_time = omp_get_wtime();

        *calc_time = (end_time - start_time) / number_of_iterations;

        if (!all_equal(array, array_length, -1, 1e-10))
        {
            fprintf(stdout, "Array not updated on host device!\n");
            fprintf(stdout, "ARRAY[0] = %f -- expected %f\n");
            fprintf(stdout, "Exiting...\n", array[0], -1);
            return 0;
        }
    }

    return 1;
}

int time_cpu(double *array, int array_length, int number_of_iterations,
             int num_threads, double *calc_time)
{
    double start_time, end_time;

    *calc_time = 0;

    for (int iter=0; iter<number_of_iterations; iter++)
    {
        set_all_equal(array, array_length, 4);

        start_time = omp_get_wtime();

        #pragma omp parallel for
        for (int i=0; i<array_length; i++)
        {
            array[i] = -1;

            if (i == 0 && omp_get_thread_num() == 0){
                fprintf(stdout, "Num threads = %d\n", omp_get_num_threads());
            }
        }

        end_time = omp_get_wtime();

        *calc_time += (end_time - start_time) / number_of_iterations;

        if (!all_equal(array, array_length, -1, 1e-10))
        {
            fprintf(stdout, "Array not updated on host device!\n");
            fprintf(stdout, "ARRAY[0] = %f -- expected %f\n");
            fprintf(stdout, "Exiting...\n", array[0], -1);
            return 0;
        }
    }
    return 1;
}

int time_gpu(double *array, int array_length, int number_of_iterations,
             int num_teams, double *transfer_time, double *calc_time)
{
    double start_time, end_time;
    double calc_start_time, calc_end_time;

    *transfer_time = 0;
    *calc_time = 0;

    for (int iter=0; iter<number_of_iterations; iter++)
    {
        set_all_equal(array, array_length, 4);

        start_time = omp_get_wtime();

        #pragma omp target enter data map(to:array[:array_length], array_length)

        calc_start_time = omp_get_wtime();
        #pragma omp target teams distribute parallel for
        for (int i=0; i<array_length; i++)
        {
            array[i] = -1;
            if (omp_get_team_num() == 0 && i == 0){
                fprintf(stdout, "Num teams = %d\n", omp_get_num_teams());
            }
        }
        calc_end_time = omp_get_wtime();

        #pragma omp target exit data map(from:array[:array_length], array_length)

        end_time = omp_get_wtime();

        *calc_time += (calc_end_time - calc_start_time) / number_of_iterations;
        *transfer_time += (end_time - start_time - calc_end_time + calc_start_time) / number_of_iterations;

        if (!all_equal(array, array_length, -1, 1e-10))
        {
            fprintf(stdout, "Array not updated on target device!\n");
            fprintf(stdout, "ARRAY[0] = %f -- expected %f\n");
            fprintf(stdout, "Exiting...\n", array[0], -1);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[])
{

    int array_length;
    array_length = (int) pow(512, 3);

    double *input_array;
    input_array = malloc(sizeof(double) * array_length);

    int num_iter;
    num_iter = 1;

    double transfer_time_gpu, calc_time_gpu;
    int num_teams;
    num_teams = 200;
    if (!time_gpu(input_array, array_length, num_iter, num_teams, &transfer_time_gpu, &calc_time_gpu))
    {
        fprintf(stdout, "GPU timing test failed! Exiting...\n");
        exit(0);
    }

    double calc_time_cpu;
    int num_threads;
    num_threads = 4;
    if (!time_cpu(input_array, array_length, num_iter, num_threads, &calc_time_cpu))
    {
        fprintf(stdout, "CPU timing test failed! Exiting...\n");
        exit(0);
    }

    double calc_time_serial;
    if (!time_serial(input_array, array_length, num_iter, &calc_time_serial))
    {
        fprintf(stdout, "Serial timing test failed! Exiting...\n");
        exit(0);
    }

    double data_size;
    data_size = sizeof(double) * array_length + sizeof(array_length);
    data_size /= 1e6;

    fprintf(stdout, "---------------------------------\n");
    fprintf(stdout, "Average time of data transfer = %g s\n", transfer_time_gpu);
    fprintf(stdout, "  + Array length = %d elements\n", array_length);
    fprintf(stdout, "  + Size of transferred data = %g MB\n", data_size);
    fprintf(stdout, "  + Number of iterations = %d\n", num_iter);
    fprintf(stdout, "Average time of calculation (GPU n_teams = ) = %g s\n", calc_time_gpu);
    fprintf(stdout, "Average time of calculation (CPU n_threads = ) = %g s\n", calc_time_cpu);
    fprintf(stdout, "Average time of calculation (CPU serial) = %g s\n", calc_time_serial);
    fprintf(stdout, "---------------------------------\n");

    return 1;
}