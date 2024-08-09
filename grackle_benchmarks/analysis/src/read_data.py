import numpy as np
from collections import defaultdict

# Function to read timing data file for grackle benchmarks
def read_timing_data(filepath, columns, column_datatypes, skip_lines=2, delimiter="|"):
    """
        filepath (str)          -- path to timing data file
        columns (list)          -- list of columns in data file, in order
        column_datatypes (dict) -- dictionary keyed by column names,
                                    with values of the datatype that column holds,
                                    specified as a string
        skip_lines (int)        -- the number of lines to skip from the top of the
                                    data file
        delimiter (str)         -- the character separating columns in the data file
    """
    
    data = defaultdict(lambda: [])
    
    with open(filepath, "r") as fptr:
        
        for line_ind, line in enumerate(fptr.readlines()):
            
            if line_ind < skip_lines:
                continue
            
            line_data = line.split(delimiter)
            
            for value, column_name in zip(line_data, columns):
                
                dtype = column_datatypes[column_name]
                
                if dtype == "str":
                    data[column_name].append(str(value.strip()))
                elif dtype == "int":
                    data[column_name].append(int(value.strip()))
                elif dtype == "float":
                    data[column_name].append(float(value.strip()))
                else:
                    print(f"Unknown datatype >{dtype}<, exiting...")
                    exit()
            
    # Convert data to numpy arrays for ease of future manipulation
    for data_key in data.keys():
        data[data_key] = np.array(data[data_key])
        
    return data