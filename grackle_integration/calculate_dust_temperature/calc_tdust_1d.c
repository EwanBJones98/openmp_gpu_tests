#include "../grackle_fields.h"
#include "../grackle_chemistry.h"

// Function to calculate the equilibrium dust temperautre
int calc_tdust_1d(grackle_chemistry my_chemistry,
                  grackle_fields my_fields,
                  int *mask)
{
    /*
        >>>> CALCULATE EQUILIBRIUM DUST TEMPERATURE <<<<

        written by: Britton Smith
        date: February, 2011

        ported by: Ewan Jones (Fortran->C)
        date: May, 2024

        PURPOSE:
            Calculate dust temperature.

        INPUTS:
            my_chemistry - grackle_chemistry struct
            my_fields    - grackle_fields struct
            mask         - iteration mask
    */

    double pert = 1e-3;

    return 1;
}