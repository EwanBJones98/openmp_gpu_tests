import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from ctypes import sizeof, c_double

sns.set_style()
sns.set_context("paper")

class TimingsData:
    
    def __init__(self, filepath, dataset_name, loop_structure_data=False,
                 optimisation_flag=True):
            
        self.filepath            = filepath
        self.name                = dataset_name
        self.loop_structure_data = loop_structure_data
        self.optimisation_flag   = optimisation_flag
        self.data                = self.read_data(self.filepath)
    
    
    def read_data(self, filepath):
        
        #* Read data from filepath into dataframe
        column_names = ["Mode", "Optimisation flag",
                        "Primordial chemistry", "Fields i dimension",
                        "Fields j dimension", "Fields k dimension",
                        "Number of timing iterations", "Number of teams",
                        "Number of threads", "Number of threads per team",
                        "Mean time (s)", "Standard deviation (s)"]
        if self.loop_structure_data:
            loop_structure_colnames = ["Mean time one loop (s)", "Standard deviation one loop (s)",
                                        "Mean time two loops (s)", "Standard deviation two loops (s)"]
            column_names = column_names[:-2]
            column_names.pop(1)
            column_names.extend(loop_structure_colnames)
        elif not self.optimisation_flag:
            column_names.pop(1)
        
        column_types = defaultdict(lambda: "int")
        column_types["Mode"]               = "str"
        column_types["Optimisation flag"]  = "opt_flag"
        for colname in ["Mean time (s)", "Standard deviation (s)",
                        "Mean time one loop (s)", "Standard deviation one loop (s)",
                        "Mean time two loops (s)", "Standard deviation two loops (s)"]:
            column_types[colname] = "float"
        
        my_data = defaultdict(lambda: [])
        
        fptr = open(filepath, "r")
        for rownum, line in enumerate(fptr.readlines()):
            
            if rownum < 2 or line[0] == "#":
                continue
            
            for colnum, value in enumerate(line.split("|")):
                try: 
                    if column_types[column_names[colnum]] == "str":
                        value = value.strip()
                    elif column_types[column_names[colnum]] == "float":
                        value = float(value.strip())
                    elif column_types[column_names[colnum]] == "int":
                        value = int(value.strip())
                    elif column_types[column_names[colnum]] == "opt_flag":
                        value = int(value.strip()[1:])
                    
                    my_data[column_names[colnum]].append(value)
                except ValueError as error:
                    print("*** ERROR ***")
                    print(f"value = {value}")
                    print(f"column num = {colnum}")
                    print(f"column name = {column_names[colnum]}")
                    print(f"filepath = {self.filepath}")
                    raise(error)
                
        #* Add a new column to each dataset showing the total field size in gb
        def _calc_field_size(row):
            field_size = sizeof(c_double)
            for dim in ["i","j","k"]:
                field_size *= row["Fields %s dimension" % dim]
                
            return field_size / pow(1024,3)
        
        #* Return dataframe     
        data = pd.DataFrame(my_data)
        data["Field size (gb)"] = data.apply(_calc_field_size, axis=1)
        
        return data