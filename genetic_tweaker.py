# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:37:38 2012

@author: jonask
"""

from kalderstam.util.filehandling import parse_file, get_cross_validation_sets
from survival.network import build_feedforward_multilayered, risk_eval
import numpy
from survival.cox_error_in_c import get_C_index
from survival.cox_genetic import c_index_error, weighted_c_index_error
from kalderstam.neural.training.davis_genetic import train_evolutionary
#from kalderstam.neural.training.genetic import train_evolutionary
from kalderstam.matlab.matlab_functions import plot_network_weights
#from kalderstam.neural.training.committee import train_committee

import logging
import kalderstam.util.graphlogger as glogger

def main(design, **train_kwargs):
    glogger.setLoggingLevel(glogger.debug)    
    
    #FAKE
    filename = "/home/gibson/jonask/Projects/DataMaker/hard_survival_test_noisyindata.txt"
    filename_val = "/home/gibson/jonask/Projects/DataMaker/hard_survival_test_val_noisyindata.txt"
    #filename = "/home/gibson/jonask/Projects/DataMaker/hard_survival_test.txt"
    #filename_val = "/home/gibson/jonask/Projects/DataMaker/hard_survival_test_val.txt"
    columns = ('X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6',  'X7', 'X8', 'X9')
    #columns = ('X0', 'X1', 'X2', 'X3', 'X4', 'X5')
    targets = ['censnoisytime', 'event']
    #targets = ['censtime', 'event']
    #targets = ['time', 'event1']

    P, T = parse_file(filename, targetcols = targets, inputcols = columns, normalize = True, separator = '\t', use_header = True)
    Pval, Tval = parse_file(filename_val, targetcols = targets, inputcols = columns, normalize = True, separator = '\t', use_header = True)

    #--------------------------------------

    #REAL    
    #filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_the_n4369_dataset_with_logs_lymf.txt"
    #columns = ('age', 'log(1+lymfmet)', 'n_pos', 'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos', 
    #           'er_cyt_pos', 'size_gt_20', 'er_cyt_pos', 'pgr_cyt_pos')
    #targets = ['time_10y', 'event_10y']
    #P, T = parse_file(filename, targetcols = targets, inputcols = columns, normalize = True, separator = '\t', use_header = True)
    #Pval, Tval = None, None

    #--------------------------------------    
    
    print('\nIncluding columns: ' + str(columns))
    print('Target columns: ' + str(targets))

    print("\nData set:")
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))
        
    for k, v in train_kwargs.iteritems():
        print(str(k) + ": " + str(v))
        
    errorfunc = c_index_error
    
    print("\nError function: " + errorfunc.__name__)
    
    print("\nDesign: " + str(design))
    layers = []
    hidden_func = design[-1]
    for layer_size in design[:-1]:
        layers.append(layer_size)

    net = build_feedforward_multilayered(input_number = len(P[0]), hidden_numbers = layers, output_number = 1, hidden_function = hidden_func, output_function = "linear")
    #net = build_feedforward(3, len(P[0]), netsize, 1, hidden_function = hidden_func, output_function = 'linear')

    #set_specific_starting_weights(net)    
    
    best_net = train_evolutionary(net, (P, T), (Pval, Tval), binary_target = 1, error_function = c_index_error, **train_kwargs)
    
    cens_output = []
    
    results = best_net.sim(P)
    best_net.trn_set = results[:, 0] #To get rid of extra dimensions
    #Now sort the set
    best_net.trn_set = numpy.sort(best_net.trn_set)
    
    for pat in P:
        cens_output.append(risk_eval(best_net, pat))
    
    cens_output = numpy.array([[val] for val in cens_output])
    
    #Calc C-index
    c_index = get_C_index(T, cens_output)
    
    print("C-Index: {0}".format(c_index))
    
    #Plot network
    plot_network_weights(best_net)
    
    glogger.show()
    
def set_specific_starting_weights(net):
    '''
    Modify as needed
    '''
    bias_weight = 5.0
    input_weight = [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0]
    hidden_num = -2
    for node in net.hidden_nodes:
        hidden_num += 2
        for key, val in node.weights.iteritems():
            # Avoid bias
            if key == net.bias_node:
                node.weights[key] = bias_weight
                bias_weight = bias_weight * -1.0 # Give next node opposite bias
            elif key < 3:
                node.weights[key] = input_weight[hidden_num % 8]
            else:
                node.weights[key] = input_weight[(hidden_num + 1) % 8]
                
    # Set output weights to 1, and bias to zero
    for key, val in net.output_nodes[0].weights.iteritems():
        net.output_nodes[0].weights[key] = 1.0
    net.output_nodes[0].weights[net.bias_node] = 0.0
    
    plot_network_weights(net)
    glogger.show()
    
if __name__ == '__main__':
    #epochs = 1, population_size = 100, mutation_chance = 0.25, random_range = 1.0, random_mean = 1.0, top_number = 25
    #error_function = sum_squares.total_error, loglevel = None
    
    #rand_mean 0.15 with mut_chance 0.5 was good
    #rand_mean 0.07 with mut_chance 1.0 was fastest
    #I have  4 terms, so a network of 8 hidden should be able to find an exact solution.
    
    #Best params so far:
    # Davis, 1.10 at 1431
    #(5, tanh), population_size = 50, random_mean = 0.25, mutation_chance = 0.2, cross_over_chance = 0.9
    
    main((4, 'tanh'), epochs = 500, population_size = 50, top_number = 25, random_mean = 0.3, mutation_half_point = 400, mutation_chance = 0.2, loglevel = logging.INFO)