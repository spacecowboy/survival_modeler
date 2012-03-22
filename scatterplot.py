import matplotlib
matplotlib.use('GTKAgg') #Only want to save images
import matplotlib.pyplot as plt
import numpy as np
import os.path
from math import sqrt
import sys
from survival.plotting import scatter
from survival.cox_error_in_c import get_C_index
from ann.filehandling import parse_data, read_data_file


def scatterplot_files(targetfile, targetcol, eventcol, modelfile, modeloutputcol, **kwargs):
    '''
    scatterplot_files(targetfile, targetcol, eventcol, modelfile, modeloutputcol)
    
    Takes two files because the target data and model data is allowed to be in different files.
    Events are ONLY taken from target data.
    Writes two files:
        scatter_cens_targetfile_modelfile.eps
        scatter_nocens_targetfile_modelfile.eps
    '''
    
    #Calculate suitable size for the figure for use in LaTEX
    fig_width_pt = 396.0  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]
    #Update settings
    plt.rcParams['figure.figsize'] = fig_size
    #params = {'axes.labelsize': 10, 
    #          'text.fontsize': 10,
    #          'legend.fontsize': 10,
    #          'xtick.labelsize': 8,
    #          'ytick.labelsize': 8,
              #'text.usetex': True,
    #          'figure.figsize': fig_size}
    #plt.rcParams.update(params)
    
#    with open(targetfile, 'r') as f:
#        X_in = [line.split() for line in f.readlines()]
#    X_in = numpy.array(X_in)
#    X = X_in[1:, first_col]
#    X = numpy.array(X, dtype = 'float')
    
    data = np.array(read_data_file(targetfile, "\t"))
    T, t = parse_data(data, inputcols = (targetcol, eventcol), ignorerows = [0], normalize = False)
    X = T[:, 0]
    events = T[:, 1]
    
#    with open(modeloutputcol, 'r') as f:
#        Y_in = [line.split() for line in f.readlines()]
#    
#    Y_in = numpy.array(Y_in)
#    Y = Y_in[1:, second_col]
#    Y = numpy.array(Y, dtype = 'float')
    
    data = np.array(read_data_file(modelfile, "\t"))
    D, t = parse_data(data, inputcols = [modeloutputcol], ignorerows = [0], normalize = False)
    Y = D[:, 0]
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
            title = "C-Index between these files is: {0}".format(c_index),
            x_label = 'Targets',
            y_label = 'Model output',
            gridsize = 30, mincnt = 0, show_plot = False)
    #plt.xlabel(os.path.basename(sys.argv[1]) + "\nC-Index between these files is: {0}".format(c_index))
    #plt.ylabel('Correlation of ' + os.path.basename(sys.argv[2]))
    
    plt.savefig('scatter_cens_{0}_{1}.eps'.format(os.path.splitext(os.path.basename(modelfile))[0], \
                                             os.path.splitext(os.path.basename(targetfile))[0]))
                                             
    
    scatter(X, Y,
            title = "C-Index between these files is: {0}".format(c_index),
            x_label = 'Targets',
            y_label = 'Model output',
            gridsize = 30, mincnt = 0, show_plot = False)
    #plt.xlabel(os.path.basename(sys.argv[1]) + "\nC-Index between these files is: {0}".format(c_index))
    #plt.ylabel('Correlation of ' + os.path.basename(sys.argv[2]))
    
    plt.savefig('scatter_nocens_{0}_{1}.eps'.format(os.path.splitext(os.path.basename(modelfile))[0], \
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
    