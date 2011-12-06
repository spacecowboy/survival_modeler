import numpy as np
import os.path
import matplotlib
matplotlib.use('Agg') #Only want to save images
import matplotlib.pyplot as plt
import sys
from survival.plotting import scatter
from survival.cox_error_in_c import get_C_index
from kalderstam.util.filehandling import parse_data, read_data_file


def scatterplot_files(targetfile, targetcol, eventcol, modelfile, modeloutputcol):
    '''
    scatterplot_files(targetfile, targetcol, eventcol, modelfile, modeloutputcol)
    
    Takes two files because the target data and model data is allowed to be in different files.
    Events are ONLY taken from target data.
    Writes two files:
        scatter_cens_targetfile_modelfile.svg
        scatter_nocens_targetfile_modelfile.svg
    '''
    
#    with open(targetfile, 'r') as f:
#        X_in = [line.split() for line in f.readlines()]
#    X_in = numpy.array(X_in)
#    X = X_in[1:, first_col]
#    X = numpy.array(X, dtype = 'float')
    
    data = np.array(read_data_file(targetfile, "\t"))
    T, t = parse_data(data, inputcols = (targetcol, eventcol), ignorerows = [0], normalize = False)
    X = T[:, 0]
    events = T[:, 1]
    
    print X
    print events
#    with open(modeloutputcol, 'r') as f:
#        Y_in = [line.split() for line in f.readlines()]
#    
#    Y_in = numpy.array(Y_in)
#    Y = Y_in[1:, second_col]
#    Y = numpy.array(Y, dtype = 'float')
    
    data = np.array(read_data_file(modelfile, "\t"))
    D, t = parse_data(data, inputcols = [modeloutputcol], ignorerows = [0], normalize = False)
    Y = D[:, 0]
    print Y
#    if event_col is not None:
#        events = X_in[1:, event_col]
#        events = numpy.array(events, dtype = 'float')
#        print 'Using events'
#    else:
#        events = None
        
#    T = numpy.empty((len(X), 2), dtype='float')
#    T[:, 0] = X
#    T[:, 1] = events
    outputs = np.empty((len(X), 2), dtype='float')
    outputs[:, 0 ] = Y
    outputs[:, 1] = events
    c_index = get_C_index(T, outputs)
    print("C-Index between these files is: {0}".format(c_index))
    
    scatter(X, Y, events = events,
            x_label = os.path.basename(modelfile) + "\nC-Index between these files is: {0}".format(c_index),
            y_label = 'Correlation of ' + os.path.basename(targetfile),
            gridsize = 30, mincnt = 0, show_plot = False)
    #plt.xlabel(os.path.basename(sys.argv[1]) + "\nC-Index between these files is: {0}".format(c_index))
    #plt.ylabel('Correlation of ' + os.path.basename(sys.argv[2]))
    
    plt.savefig('scatter_cens_{0}_{1}.svg'.format(os.path.splitext(os.path.basename(modelfile))[0], \
                                             os.path.splitext(os.path.basename(targetfile))[0]))
                                             
    
    scatter(X, Y,
            x_label = os.path.basename(modelfile) + "\nC-Index between these files is: {0}".format(c_index),
            y_label = 'Output of ' + os.path.basename(targetfile),
            gridsize = 30, mincnt = 0, show_plot = False)
    #plt.xlabel(os.path.basename(sys.argv[1]) + "\nC-Index between these files is: {0}".format(c_index))
    #plt.ylabel('Correlation of ' + os.path.basename(sys.argv[2]))
    
    plt.savefig('scatter_nocens_{0}_{1}.svg'.format(os.path.splitext(os.path.basename(modelfile))[0], \
                                             os.path.splitext(os.path.basename(targetfile))[0]))
    
    #plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('Not enough arguments. Takes two files! Targetfile, Modelfile')

    if len(sys.argv) >= 4:
        target_col = sys.argv[3]
    else:
        target_col = 0
    
    if len(sys.argv) >= 5:
        model_col = sys.argv[4]
    else:
        model_col = 1
    
    if len(sys.argv) >= 6:
        event_col = sys.argv[5]
    else:
        event_col = 2
        
    scatterplot_files(sys.argv[1], target_col, event_col, sys.argv[2], model_col)
    