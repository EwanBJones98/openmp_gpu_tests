#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
    #include <omp.h>
#endif


// >>> Define structure to hold fields <<<
typedef struct
{
    int length;

    double *density;
    double *internal_energy;
    double *pressure;
    double *temperature;
    double *h2_fraction;
} field_data;
// >>> --- <<<

// >>> Convenience functions <<<
void set_array(double *array, int array_length, double value)
{
    for (int i=0; i<array_length; i++)
    {
        array[i] = value;
    }
}
// >>> --- <<<

// >>> Define field enumerator and handling functions <<<
enum field
{
    DENSITY,
    INTERNAL_ENERGY,
    PRESSURE,
    TEMPERATURE,
    H2_FRACTION
};

double* get_field(enum field requested_field, field_data *my_fields)
{
    switch (requested_field)
    {
        case (DENSITY):
            return my_fields->density;
        case (INTERNAL_ENERGY):
            return my_fields->internal_energy;
        case (PRESSURE):
            return my_fields->pressure;
        case (TEMPERATURE):
            return my_fields->temperature;
        case (H2_FRACTION):
            return my_fields->h2_fraction;
        default:
            fprintf(stdout, "Field not recognised! Exiting...\n");
            exit(0);
    }
}

void init_fields(field_data *my_fields, int field_length, double init_value)
{
    my_fields->length = field_length;

    my_fields->density         = malloc(sizeof(double) * my_fields->length);
    my_fields->internal_energy = malloc(sizeof(double) * my_fields->length);
    my_fields->pressure        = malloc(sizeof(double) * my_fields->length);
    my_fields->temperature     = malloc(sizeof(double) * my_fields->length);
    my_fields->h2_fraction     = malloc(sizeof(double) * my_fields->length);

    set_array(my_fields->density, my_fields->length, init_value);
    set_array(my_fields->internal_energy, my_fields->length, init_value);
    set_array(my_fields->pressure, my_fields->length, init_value);
    set_array(my_fields->temperature, my_fields->length, init_value);
    set_array(my_fields->h2_fraction, my_fields->length, init_value);
}
// >>> --- <<<

// >>> Custom omp mapping functions <<<
int update_target_field(int mode, enum field requested_field, field_data *host_fields, field_data *target_fields)
{
    // mode = 1 -> map data from the host to the target
    // mode = 2 -> map data from the target to the host
    void *destination_field, *source_field;
    int destination_ID, source_ID;

    if (mode == 1)
    {
        source_field      = get_field(requested_field, host_fields);
        destination_field = get_field(requested_field, target_fields);

        source_ID      = omp_get_initial_device();
        destination_ID = omp_get_default_device();
    } else if (mode == 2) {
        source_field      = get_field(requested_field, target_fields);
        destination_field = get_field(requested_field, host_fields);

        source_ID      = omp_get_default_device();
        destination_ID = omp_get_initial_device();
    } else {
        fprintf(stderr, "mode = %d is not a valid option for function `update_target_field`. Exiting...\n", mode);
        exit(1);
    }

    size_t field_bytes = sizeof(double) * host_fields->length;
    fprintf(stdout, "dest=%s\nsource=%s\n", destination_field, source_field);
    fprintf(stdout, "destID=%d\nsourceID=%d\n", destination_ID, source_ID);
    return omp_target_memcpy(destination_field, source_field, field_bytes, 0,  0, destination_ID, source_ID);
}

void field_update_to(enum field active_fields[], int num_active_fields, field_data *host_fields, field_data *target_fields)
{
    for (int i=0; i<num_active_fields; i++)
    {
        if (update_target_field(1, active_fields[i], host_fields, target_fields))
        {
            fprintf(stderr, "Copying active field number %d to target device failed! Exiting...\n", i);
        }
    }
}

void field_update_from(enum field active_fields[], int num_active_fields, field_data *host_fields, field_data *target_fields)
{
    for (int i=0; i<num_active_fields; i++)
    {
        if (update_target_field(2, active_fields[i], host_fields, target_fields))
        {
            fprintf(stderr, "Copying active field number %d from target device failed! Exiting...\n", i);
        }
    }
}

void alloc_target_fields(enum field active_fields[], int num_active_fields, field_data *target_fields, int field_length)
{
    int target_ID = omp_get_default_device();
    size_t array_bytes = sizeof(double) * field_length;
    double *current_field;

    for (int i=0; i<num_active_fields; i++)
    {
        current_field = get_field(active_fields[i], target_fields);
        current_field = omp_target_alloc(array_bytes, target_ID);
    }
}

void free_target_fields(enum field active_fields[], int num_active_fields, field_data *target_fields)
{
    int target_ID = omp_get_default_device();
    
    for (int i=0; i<num_active_fields; i++)
    {
        omp_target_free(get_field(active_fields[i], target_fields), target_ID);
    }
}
// >>> --- <<<


int main(int argc, char *argv[])
{
    field_data my_fields;
    init_fields(&my_fields, 10000, -1);

    enum field active_fields[] = {DENSITY, H2_FRACTION};
    int num_active_fields          = 2;

    field_data my_target_fields;
    alloc_target_fields(active_fields, num_active_fields, &my_target_fields, my_fields.length);
    field_update_to(active_fields, num_active_fields, &my_fields, &my_target_fields);

    double *density_ptr = my_target_fields.density;
    #pragma omp target is_device_ptr(density_ptr)
    {
        for (int i=0; i<my_fields.length; i++)
        {
            my_target_fields.density[i] = 44;
        }
        
    }

    fprintf(stdout, "pre-update: %f\n", my_target_fields.density[0]);

    field_update_from(active_fields, num_active_fields, &my_fields, &my_target_fields);

    fprintf(stdout, "post-update: %f\n", my_target_fields.density[0]);

    free_target_fields(active_fields, num_active_fields, &my_target_fields);

    return 0;
}