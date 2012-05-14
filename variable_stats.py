#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage: program.py [options] arguments

Options:
-h --help          show this help message and exit

'''
from __future__ import division
import sys
import numpy as np
from ann.filehandling import read_data_file, parse_headers_in_file

def main(datafile):
    print("\nFile: " + datafile)
    data = read_data_file(datafile, separator=',')
    # data is now a python listlist of strings
    # remove headers and convert to numpy
    data = np.array(data[1:])
    # read headers
    column_map = parse_headers_in_file(datafile)

    #Now report on missing data
    print("{:20s} : {:5s} percentage".format("Covariate", "num"))
    for var, col in column_map.iteritems():
        missing = len(data[:, col][data[:, col] == ''])
        print("{:20s} : {:5s} {:.2f}%".format(var, str(missing), 100*missing/len(data[:, col])))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exit(__doc__)

    testfile = '/home/gibson/jonask/DataSets/breast_cancer_1/n4369_targetthird.csv'
    trnfile = '/home/gibson/jonask/DataSets/breast_cancer_1/n4369_trainingtwothirds.csv'
    files = [trnfile, testfile]

    for datafile in files:
        main(datafile)
