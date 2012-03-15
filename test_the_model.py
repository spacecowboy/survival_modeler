# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:20:54 2012

@author: jonask
"""

from model_tester import test_model_arrays
from kalderstam.util.normalizer import normalizeArrayLike
from kalderstam.util.filehandling import parse_file

def main():
    pass

if __name__ == "__main__":
    #Test the model on the test data!
    model = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/publication_data/ann/cens_10y/2_tanh_1328829716.pcom'
    testdata = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/The_remaining_third_of_the_n4369_dataset.txt_with_logs'
    columns = ['age', 'log(1+lymfmet)', 'n_pos', 'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos', 
               'er_cyt_pos', 'size_gt_20', 'er_cyt', 'pgr_cyt']
    
    #targets = ['time_10y', 'event_10y']
    trainingdata = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_the_n4369_dataset_with_logs_lymf.txt'
    
    print("Retrieving training data...")
    # Normalize the test data as we normalized the training data
    normP, bah = parse_file(trainingdata, inputcols = columns, normalize = False, separator = '\t', 
                      use_header = True)
    
    print("Retrieving test data...")
    unNormedTestP, uT = parse_file(testdata, inputcols = columns, normalize = False, separator = '\t',
                      use_header = True)
                      
    print("Normalizing test data...")
    P = normalizeArrayLike(unNormedTestP, normP)
    
    print("Getting outputs for test data...")
    #Wihtout targets, we only get the outputs
    outputs = test_model_arrays(model, testdata, P, None)
    print("We have outputs! Length: {}".format(len(outputs)))
    
    #model_output_file = test_model(model, testdata, None, None, *columns)
    #scatterplot_files(model_output_file, 0, 2, model_output_file, 1)