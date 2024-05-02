#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
    #include <omp.h>
#endif


int main(int argc, char* argv[])
{

    double x[10];

    #pragma omp target map(tofrom:x)
    {
        for (int i=0; i<10; i++)
        {
            x[i] *= 2;
        }
    }

    for (int i=0; i<10; i++)
    {
        fprintf(stdout, "%g", x[i]);
    }

    return 1;
}