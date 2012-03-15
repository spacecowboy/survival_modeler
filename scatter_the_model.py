# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 12:39:04 2012

@author: jonask
"""

from model_tester import test_model
from scatterplot import scatterplot_files

def main():
    pass

if __name__ == "__main__":
    #Test the model on the test data!
    model = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/publication_data/ann/cens_10y/2_tanh_1328829716.pcom'
    testdata = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/The_remaining_third_of_the_n4369_dataset_with_logs_and_targets.txt'
    columns = ['age', 'log(1+lymfmet)', 'n_pos', 'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos', 
               'er_cyt_pos', 'size_gt_20', 'er_cyt', 'pgr_cyt']
    
    targets = ['time_10y', 'event_10y']
    trainingdata = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_the_n4369_dataset_with_logs_lymf.txt'

    #Scatter training data
    model_output_file = test_model(model, trainingdata, targets[0], targets[1], time_step_size = 2, *columns)
    scatterplot_files(model_output_file, 0, 2, model_output_file, 1)    
    
    #Scatter test data
    model_output_file = test_model(model, testdata, targets[0], targets[1], time_step_size = 2, *columns)
    scatterplot_files(model_output_file, 0, 2, model_output_file, 1)  