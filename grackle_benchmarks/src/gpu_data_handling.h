#ifndef _GPU_DATA_HANDLING_H
#define _GPU_DATA_HANDLING_H

void enter_gpu(double *my_field_buffer,
                chemistry_data *my_chemistry,
                grackle_field_data *my_fields,
                code_units *my_units);

void exit_gpu(double *my_field_buffer,
                chemistry_data *my_chemistry,
                grackle_field_data *my_fields,
                code_units *my_units);

#endif