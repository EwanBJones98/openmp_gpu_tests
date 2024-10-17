#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utility.h"

void print_linenumber(int line_number, char *filename)
{
    fprintf(stdout, "-- Reached %s:%d --\n", filename, line_number);
    fflush(stdout);
}