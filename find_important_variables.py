#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage: program.py [options] arguments

This script takes a trained model and tries to find the variables that are important.
It does so by replacing the test data's variables with the training set's averages,
one by one. Effectively this sets the variable to zero after normalization.
By removing the variable from the data set, it then calculates what the drop
(or gain) in c-index is by the loss of this information.

You need:
A trained model
The training data
The test data
Covariate columns

Output is a list, sorted by drop in c-index, biggest drop is most important variable.

Options:
-h --help          show this help message and exit

'''

import sys, pickle
import numpy as np
from model_tester import test_model, test_model_arrays
from ann.filehandling import parse_file, parse_headers_in_file, normalizeArrayLike
from survival.cox_error_in_c import get_C_index

def main(model, test_data, test_targets, column_map):
    print(column_map)
    #First establish baseline c-index
    out = np.array([[model.risk_eval(inputs)] for inputs in test_data])
    base_cindex = get_C_index(test_targets, out)
    #Now we can calculate any changes. Do so now for each variable
    #TODO: make sure they are ordered correctly
    variable_changes = {}
    for var, i in column_map.iteritems():
        print("Checking {}, {}".format(i, var))
        #Make a copy of the data set so we can modify the variable
        temp_data = test_data.copy()
        #Set this variable to zero
        temp_data[:,i] = 1
        #Generate output and calc c-index. Also increase by 100
        out = np.array([[model.risk_eval(inputs)] for inputs in temp_data])
        variable_changes[var] = 100*(base_cindex - get_C_index(test_targets, out))

    #All variables completed. Return dictionary
    return variable_changes

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exit(__doc__)

    testfile = '/home/gibson/jonask/DataSets/breast_cancer_1/n4369_targetthird.csv'
    trnfile = '/home/gibson/jonask/DataSets/breast_cancer_1/n4369_trainingtwothirds.csv'
    columns = ['age', 'log(1+lymfmet)', 'n_pos', 'tumsize',
               'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos',
               'er_cyt_pos', 'size_gt_20', 'er_cyt', 'pgr_cyt']
    targets = ['time_10y', 'event_10y']

    # Normalize the test data as we normalized the training data
    normP, bah = parse_file(trnfile, inputcols = columns,
                            targetcols = targets, normalize = False, separator = ',',
                            use_header = True)

    unNormedTestP, test_targets = parse_file(testfile, inputcols = columns,
                                  targetcols = targets, normalize = False,
                                  separator = ',', use_header = True)

    test_data = normalizeArrayLike(unNormedTestP, normP)

    # Read the model from file
    savefile = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/publication_data/ann/cens_10y/2_tanh_1328829716.pcom'

    with open(savefile, 'r') as FILE:
        model = pickle.load(FILE)

    # Get a proper header map
    column_map = parse_headers_in_file(columns, testfile)

    # Explore variable changes
    variable_changes = main(model, test_data, test_targets, column_map)

    #Print results, sort by change
    print("\nCovariates, sorted by importance:")
    for var, val in sorted(variable_changes.items(), lambda x, y: cmp(abs(y[1]), abs(x[1]))):
        print("{:15s}: {:.3f}".format(var, val))
