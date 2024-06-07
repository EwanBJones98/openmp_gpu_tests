/***********************************************************************
/
/ Initialize chemistry and cooling rate data
/
/
/ Copyright (c) 2013, Enzo/Grackle Development Team.
/
/ Distributed under the terms of the Enzo Public Licence.
/
/ The full license is in the file LICENSE, distributed with this
/ software.
************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "grackle_macros.h"
#include "grackle_types.h"
#include "grackle_chemistry_data.h"
#include "phys_constants.h"
#ifdef _OPENMP
#include <omp.h>
#endif

// extern int grackle_verbose;

void show_parameters(FILE *fp, chemistry_data *my_chemistry);

int local_free_chemistry_data(chemistry_data *my_chemistry, chemistry_data_storage *my_rates);

int initialize_rates(chemistry_data *my_chemistry, chemistry_data_storage *my_rates, code_units *my_units,
                double co_length_units, double co_density_units);

/**
 * Initializes an empty #chemistry_data_storage struct with zeros and NULLs.
 */
void initialize_empty_chemistry_data_storage_struct(chemistry_data_storage *my_rates)
{
  my_rates->k1 = NULL;
  my_rates->k2 = NULL;
  my_rates->k3 = NULL;
  my_rates->k4 = NULL;
  my_rates->k5 = NULL;
  my_rates->k6 = NULL;
  my_rates->k7 = NULL;
  my_rates->k8 = NULL;
  my_rates->k9 = NULL;
  my_rates->k10 = NULL;
  my_rates->k11 = NULL;
  my_rates->k12 = NULL;
  my_rates->k13 = NULL;
  my_rates->k14 = NULL;
  my_rates->k15 = NULL;
  my_rates->k16 = NULL;
  my_rates->k17 = NULL;
  my_rates->k18 = NULL;
  my_rates->k19 = NULL;
  my_rates->k20 = NULL;
  my_rates->k21 = NULL;
  my_rates->k22 = NULL;
  my_rates->k23 = NULL;
  my_rates->k13dd = NULL;
  my_rates->k24 = 0.;
  my_rates->k25 = 0.;
  my_rates->k26 = 0.;
  my_rates->k27 = 0.;
  my_rates->k28 = 0.;
  my_rates->k29 = 0.;
  my_rates->k30 = 0.;
  my_rates->k31 = 0.;
  my_rates->k50 = NULL;
  my_rates->k51 = NULL;
  my_rates->k52 = NULL;
  my_rates->k53 = NULL;
  my_rates->k54 = NULL;
  my_rates->k55 = NULL;
  my_rates->k56 = NULL;
  my_rates->k57 = NULL;
  my_rates->k58 = NULL;
  my_rates->h2dust = NULL;
  my_rates->n_cr_n = NULL;
  my_rates->n_cr_d1 = NULL;
  my_rates->n_cr_d2 = NULL;
  my_rates->ceHI = NULL;
  my_rates->ceHeI = NULL;
  my_rates->ceHeII = NULL;
  my_rates->ciHI = NULL;
  my_rates->ciHeI = NULL;
  my_rates->ciHeIS = NULL;
  my_rates->ciHeII = NULL;
  my_rates->reHII = NULL;
  my_rates->reHeII1 = NULL;
  my_rates->reHeII2 = NULL;
  my_rates->reHeIII = NULL;
  my_rates->brem = NULL;
  my_rates->comp = 0.;
  my_rates->comp_xray = 0.;
  my_rates->temp_xray = 0.;
  my_rates->piHI = 0.;
  my_rates->piHeI = 0.;
  my_rates->piHeII = 0.;
  my_rates->crsHI = 0.;
  my_rates->crsHeI = 0.;
  my_rates->crsHeII = 0.;
  my_rates->hyd01k = NULL;
  my_rates->h2k01 = NULL;
  my_rates->vibh = NULL;
  my_rates->roth = NULL;
  my_rates->rotl = NULL;
  my_rates->GP99LowDensityLimit = NULL;
  my_rates->GP99HighDensityLimit = NULL;
  my_rates->GAHI = NULL;
  my_rates->GAH2 = NULL;
  my_rates->GAHe = NULL;
  my_rates->GAHp = NULL;
  my_rates->GAel = NULL;
  my_rates->H2LTE = NULL;
  my_rates->HDlte = NULL;
  my_rates->HDlow = NULL;
  my_rates->cieco = NULL;
  my_rates->gammah = 0.;
  my_rates->regr = NULL;
  my_rates->gamma_isrf = 0.;
  my_rates->gas_grain = NULL;
  my_rates->cloudy_data_new = -1;
}

int local_initialize_chemistry_data(chemistry_data *my_chemistry,
                                    chemistry_data_storage *my_rates,
                                    code_units *my_units)
{

  /* Better safe than sorry: Initialize everything to NULL/0 */
  initialize_empty_chemistry_data_storage_struct(my_rates);

  // Activate dust chemistry machinery.
  if (my_chemistry->dust_chemistry > 0) {

    if (my_chemistry->metal_cooling < 1) {
      fprintf(stderr, "ERROR: dust_chemistry > 0 requires metal_cooling > 0.\n");
      return FAIL;
    }

    if (my_chemistry->photoelectric_heating < 0) {
      my_chemistry->photoelectric_heating = 2;
    }

    if (my_chemistry->dust_recombination_cooling < 0) {
      my_chemistry->dust_recombination_cooling = 1;
    }

    if (my_chemistry->primordial_chemistry > 1 &&
        my_chemistry->h2_on_dust == 0) {
      my_chemistry->h2_on_dust = 1;
    }

  }

  // Default photo-electric heating to off if unset.
  if (my_chemistry->photoelectric_heating < 0) {
    my_chemistry->photoelectric_heating = 0;
  }

//initialize OpenMP
# ifndef _OPENMP
  if (my_chemistry->omp_nthreads > 1) {
    fprintf(stdout,
            "omp_nthreads can't be set when Grackle isn't compiled with "
            "OPENMP\n");
    return FAIL;
  }
# else _OPENMP
  if (my_chemistry->omp_nthreads < 1) {
    // this is the default behavior (unless the user intervenes)
    my_chemistry->omp_nthreads = omp_get_max_threads();
  }
//number of threads
  omp_set_num_threads( my_chemistry->omp_nthreads );

//schedule
//const int chunk_size = -1;  // determined by default
  const int chunk_size = 1;

//omp_set_schedule( omp_sched_static,  chunk_size );
//omp_set_schedule( omp_sched_dynamic, chunk_size );
  omp_set_schedule( omp_sched_guided,  chunk_size );
//omp_set_schedule( omp_sched_auto,    chunk_size );
# endif

  /* Only allow a units to be one with proper coordinates. */
  if (my_units->comoving_coordinates == FALSE &&
      my_units->a_units != 1.0) {
    fprintf(stderr, "ERROR: a_units must be 1.0 if comoving_coordinates is 0.\n");
    return FAIL;
  }

  if (my_chemistry->primordial_chemistry == 0) {
    /* In fully tabulated mode, set H mass fraction according to
       the abundances in Cloudy, which assumes n_He / n_H = 0.1.
       This gives a value of about 0.716. Using the default value
       of 0.76 will result in negative electron densities at low
       temperature. Below, we set X = 1 / (1 + m_He * n_He / n_H). */
    my_chemistry->HydrogenFractionByMass = 1. / (1. + 0.1 * 3.971);
  }

  double co_length_units, co_density_units;
  if (my_units->comoving_coordinates == TRUE) {
    co_length_units = my_units->length_units;
    co_density_units = my_units->density_units;
  }
  else {
    co_length_units = my_units->length_units *
      my_units->a_value * my_units->a_units;
    co_density_units = my_units->density_units /
      POW(my_units->a_value * my_units->a_units, 3);
  }

  //* Call initialise_rates to compute rate tables.
  initialize_rates(my_chemistry, my_rates, my_units, co_length_units, co_density_units);

  return SUCCESS;
}

// Define helpers for the show_parameters function
// NOTE: it's okay that these functions all begin with an underscore since they
//       each have internal linkage (i.e. they are each declared static)
static void _show_field_INT(FILE *fp, const char* field, int val)
{ fprintf(fp, "%-33s = %d\n", field, val); }
static void _show_field_DOUBLE(FILE *fp, const char* field, double val)
{ fprintf(fp, "%-33s = %g\n", field, val); }
static void _show_field_STRING(FILE *fp, const char* field, const char* val)
{ fprintf(fp, "%-33s = %s\n", field, val); }

// this function writes each field of my_chemistry to fp
void show_parameters(FILE *fp, chemistry_data *my_chemistry){
  #define ENTRY(FIELD, TYPE, DEFAULT_VAL) \
    _show_field_ ## TYPE (fp, #FIELD, my_chemistry->FIELD);
  #include "grackle_chemistry_data_fields.def"
  #undef ENTRY
}

int local_free_chemistry_data(chemistry_data *my_chemistry,
                              chemistry_data_storage *my_rates) {
  if (my_chemistry->primordial_chemistry > 0) {
    GRACKLE_FREE(my_rates->ceHI);
    GRACKLE_FREE(my_rates->ceHeI);
    GRACKLE_FREE(my_rates->ceHeII);
    GRACKLE_FREE(my_rates->ciHI);
    GRACKLE_FREE(my_rates->ciHeI);
    GRACKLE_FREE(my_rates->ciHeIS);
    GRACKLE_FREE(my_rates->ciHeII);
    GRACKLE_FREE(my_rates->reHII);
    GRACKLE_FREE(my_rates->reHeII1);
    GRACKLE_FREE(my_rates->reHeII2);
    GRACKLE_FREE(my_rates->reHeIII);
    GRACKLE_FREE(my_rates->brem);
    GRACKLE_FREE(my_rates->hyd01k);
    GRACKLE_FREE(my_rates->h2k01);
    GRACKLE_FREE(my_rates->vibh);
    GRACKLE_FREE(my_rates->roth);
    GRACKLE_FREE(my_rates->rotl);
    GRACKLE_FREE(my_rates->GP99LowDensityLimit);
    GRACKLE_FREE(my_rates->GP99HighDensityLimit);

    GRACKLE_FREE(my_rates->HDlte);
    GRACKLE_FREE(my_rates->HDlow);
    GRACKLE_FREE(my_rates->cieco);
    GRACKLE_FREE(my_rates->GAHI);
    GRACKLE_FREE(my_rates->GAH2);
    GRACKLE_FREE(my_rates->GAHe);
    GRACKLE_FREE(my_rates->GAHp);
    GRACKLE_FREE(my_rates->GAel);
    GRACKLE_FREE(my_rates->H2LTE);
    GRACKLE_FREE(my_rates->gas_grain);

    GRACKLE_FREE(my_rates->k1);
    GRACKLE_FREE(my_rates->k2);
    GRACKLE_FREE(my_rates->k3);
    GRACKLE_FREE(my_rates->k4);
    GRACKLE_FREE(my_rates->k5);
    GRACKLE_FREE(my_rates->k6);
    GRACKLE_FREE(my_rates->k7);
    GRACKLE_FREE(my_rates->k8);
    GRACKLE_FREE(my_rates->k9);
    GRACKLE_FREE(my_rates->k10);
    GRACKLE_FREE(my_rates->k11);
    GRACKLE_FREE(my_rates->k12);
    GRACKLE_FREE(my_rates->k13);
    GRACKLE_FREE(my_rates->k13dd);
    GRACKLE_FREE(my_rates->k14);
    GRACKLE_FREE(my_rates->k15);
    GRACKLE_FREE(my_rates->k16);
    GRACKLE_FREE(my_rates->k17);
    GRACKLE_FREE(my_rates->k18);
    GRACKLE_FREE(my_rates->k19);
    GRACKLE_FREE(my_rates->k20);
    GRACKLE_FREE(my_rates->k21);
    GRACKLE_FREE(my_rates->k22);
    GRACKLE_FREE(my_rates->k23);
    GRACKLE_FREE(my_rates->k50);
    GRACKLE_FREE(my_rates->k51);
    GRACKLE_FREE(my_rates->k52);
    GRACKLE_FREE(my_rates->k53);
    GRACKLE_FREE(my_rates->k54);
    GRACKLE_FREE(my_rates->k55);
    GRACKLE_FREE(my_rates->k56);
    GRACKLE_FREE(my_rates->k57);
    GRACKLE_FREE(my_rates->k58);
    GRACKLE_FREE(my_rates->h2dust);
    GRACKLE_FREE(my_rates->n_cr_n);
    GRACKLE_FREE(my_rates->n_cr_d1);
    GRACKLE_FREE(my_rates->n_cr_d2);
  }

  return SUCCESS;
}
