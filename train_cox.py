# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:48:00 2011

@author: jonask
"""

import sys
from cox_model.cox_trainer import train_model
from model_tester import test_model

if __name__ == '__main__':
#    if len(sys.argv) < 5:
#        print('Proper usage is: {0} datafile, inputcolumns, targetcolumn, eventcolumn)'.format(sys.argv[0])
#        sys.exit

    filename = "/home/gibson/jonask/DataSets/breast_cancer_1/n4369_trainingtwothirds.csv"
    testfilename = None
    testtargets = []
    columns = ('age', 'log(1+lymfmet)', 'n_pos', 'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos',
               'er_cyt_pos', 'size_gt_20', 'er_cyt', 'pgr_cyt')
    targets = ['time_10y', 'event_10y']
    separator = ','

    #filename = "/home/gibson/jonask/DataSets/breast_cancer_1/squares_noisyindata.csv"
    #testfilename = "/home/gibson/jonask/DataSets/breast_cancer_1/squares_test_noisyindata.csv"
    #separator = ','
    #columns = ('X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9')
    #targets = ['censnoisytime', 'event']
    #testtargets = ['time', 'event1']


    #filename = "/home/gibson/jonask/Projects/DataSets/bsi_localized.csv"
    #separator = ','
    #testfilename = None
    #columns = ('Age', 'BSI', 'nMet', 'nAreas', 'BSI_A1', 'BSI_A2', 'BSI_A3', 'BSI_A4', 'BSI_A5', 'BSI_A6',
    #            'BSI_A7', 'BSI_A8', 'BSI_A9', 'BSI_A10', 'BSI_A11', 'BSI_A12', 'N_A1', 'N_A2', 'N_A3',
    #             'N_A4', 'N_A5', 'N_A6', 'N_A7', 'N_A8', 'N_A9', 'N_A10', 'N_A11', 'N_A12')
    #targets = ['Stid', 'Event']
    #testtargets = []

    print("\nTraining a cox committee...")

    comsize = 1
    if len(sys.argv) > 1:
        comsize = int(sys.argv[1])

    model_file = train_model(filename, columns, targets, separator = separator, comsize=comsize)

    print("\nProceeding with plotting on training data...")

    model_output_file = test_model(model_file, filename, targets[0], targets[1], separator, *columns)

    print("\nModel output stored in {0}".format(model_output_file))

    #Against the same file, just another column
    #scatterplot_files(model_output_file, 0, 2, model_output_file, 1)

    cmd = 'python model_tester.py "{model_file}" YOUR_TEST_FILE_HERE "{0}" "{1}" "{2}"'.format(separator, targets[0], targets[1], \
                                                                                model_file = model_file)
    selfcmd = ''
    if testfilename is not None:
        selfcmd = 'python model_tester.py "{model_file}" "{testfile}" "{0}" "{1}" "{2}"'.format(separator, testtargets[0], testtargets[1], \
                                                                        model_file = model_file, testfile = testfilename)
    for col in columns:
        cmd += ' "{0}"'.format(col)
        if testfilename is not None:
            selfcmd += ' "{0}"'.format(col)

    print('''
    Completed.
    If you wish to test your model against some test data, use this command:
        {0}
    To get scatter plots for test data, run this command:
        {1}'''.format(cmd, selfcmd))
