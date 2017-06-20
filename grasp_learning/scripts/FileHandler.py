#!/usr/bin/env python

import numpy as np
import os

def create_directory(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

def create_file(filename):
    reset_files([filename])
    return filename 
# Opens and closes a file to empty its content
def reset_files(file_names):
    for file in file_names:
        open(file, 'w').close()

def save_data_to_file(filename, data):

    f_handle = file(filename,'a')
    np.savetxt(f_handle, data, delimiter='\t')
    f_handle.close()

def save_matrix_data_to_file(filename, data):
    f_handle = file(filename,'a')
    for inner_list in data:
        for elem in inner_list:
            f_handle.write(str(elem)+" ")
    f_handle.write("\n")
    f_handle.close()

def save_vector_data_to_file(filename, data):
    f_handle = file(filename,'a')
    for elem in data:
        f_handle.write(str(elem)+" ")
    f_handle.write("\n")
    f_handle.close()

def read_csv_file(filename):
    params = []
    with open(filename, "rb") as f:
        reader = csv.reader(f)
        for row in reader:
            params.append(row)
    return params

def parse_input_output_data(input_file):
    input_data = []
    output_data = []
    input_ = []
    output_ = []
    i = 0
    with open(input_file, 'rU') as f:
        for line in f:
            #Skip first two lines of the file
            if (i==0 or i==1):
                i+=1
                continue
            line = line.split()
            for string in xrange(len(line)):
                if string%2==0:
                    input_data.append(float(line[string])+np.random.normal(0, 0.1))
                else:
                    output_data.append(float(line[string]))

            input_.append(input_data)
            output_.append(output_data)
            input_data = []
            output_data = []

    return np.asarray(input_), np.asarray(output_)
