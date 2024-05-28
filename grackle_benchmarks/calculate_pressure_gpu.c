#include "calculate_pressure.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

void enter_calculate_pressure()
{
    #pragma omp target enter data map(alloc:)

    #pragma omp target update to()
}   

void exit_calculate_pressure()
{
    #pragma omp target update from()

    #pragma omp target exit data map(delete:)
}