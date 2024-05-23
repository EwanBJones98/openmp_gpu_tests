#ifndef _GRACKLE_FIELDS
#define _GRACKLE_FIELDS

typedef struct
{
    int grid_rank;
    int *grid_dimensions;
    int *grid_start;
    int *grid_end;

    double *isrf_habing;

    double *density;
    double *pressure;
    double *internal_energy;
    double *temperature;
    double *metallicity;

    double *dust_density;
    double *dust_temperature;
    double *dust_metallicity;

    double *e_density;
    double *HI_density;
    double *HII_density;
    double *HeI_density;
    double *HeII_density;
    double *HeIII_density;
    double *HM_density;
    double *H2I_density;
    double *H2II_density;
} grackle_fields;

#endif