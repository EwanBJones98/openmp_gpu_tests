typedef struct
{   
    int grid_rank;
    int field_size; //! THIS IS AN ADDITION TO THE STANDARD GRACKLE STRUCT
    int *grid_dimension;
    int *grid_start;
    int *grid_end;

    double* density;
    double* internal_energy;
    double* pressure;
} grackle_fields;

typedef struct
{
    double gamma;
} grackle_chemistry;

typedef struct
{
  int i_start;
  int i_end;
  int i_dim;

  int j_start;
  int j_dim;
  /* j_end isn't needed */

  int k_start;
  /* k_end & k_dim aren't needed */

  int num_j_inds;
  int outer_ind_size;

} grackle_index_helper;

typedef struct
{
  int start;
  int end;
} grackle_index_range;

grackle_index_helper _build_index_helper(const int grid_rank, const int grid_dimension[3],
                        const int grid_start[3], const int grid_end[3])
{
  grackle_index_helper out;
  const int rank = grid_rank;

  /* handle i indices */
  out.i_dim   = grid_dimension[0];
  out.i_start = grid_start[0];
  out.i_end   = grid_end[0];

  /* handle j indices (j_end isn't tracked by grackle_index_helper) */
  out.j_dim   = (rank >= 2) ? grid_dimension[1] : 1;
  out.j_start = (rank >= 2) ? grid_start[1]     : 0;
  int j_end   = (rank >= 2) ? grid_end[1]       : 0;
  out.num_j_inds = (j_end - out.j_start) + 1;

  /* handle k indices (k_end & k_dim aren't tracked by grackle_index_helper) */
  out.k_start = (rank >= 3) ? grid_start[2]     : 0;
  int k_end   = (rank >= 3) ? grid_end[2]       : 0;
  int num_k_inds = (k_end - out.k_start) + 1;

  out.outer_ind_size = num_k_inds * out.num_j_inds;
  return out;
}

static inline grackle_index_range _inner_range(int outer_index, 
                                               const grackle_index_helper* ind_helper)
{
  int k = (outer_index / ind_helper->num_j_inds) + ind_helper->k_start;
  int j = (outer_index % ind_helper->num_j_inds) + ind_helper->j_start;
  int outer_offset = ind_helper->i_dim * (j + ind_helper->j_dim * k);
  grackle_index_range out = {ind_helper->i_start + outer_offset,
                             ind_helper->i_end + outer_offset};
  return out;
}