import pickle, os.path
from math import sqrt
try:
    import matplotlib
    matplotlib.use('GTKAgg') #Only want to save images
    import matplotlib.pyplot as plt
except ImportError:
    plt = None #This makes matplotlib optional
from survival.cox_error_in_c import get_C_index
from kalderstam.util.filehandling import parse_file
from survival.plotting import kaplanmeier
import numpy

def test_model(savefile, filename, targetcol, eventcol, *cols):
    '''
    test_model(savefile, filename, targetcol, eventcol, *cols)
    
    Runs the model on the test data and outputs the result together with a Kaplan Meier plot of two groups of equal
    size.
    Saves a Kaplan-Meir plot as kaplanmeier_savefile_filename.eps
    Saves a file of the model output as .savefile_test_filename.cvs and returns this filename of the structure:
        Targets\tOutputs\tEvents
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
    
    
    headers = False

    targets = [targetcol, eventcol]
    columns = tuple(cols) #the rest
    try:
        float(columns[0])
        targets = [int(x) for x in targets]
        columns = tuple([int(x) for x in columns])
    except:
        headers = True #Because the items in columns can not be numbers
        print("Using headers")
    print('Using file: {0}'.format(filename))

    with open(savefile, 'r') as FILE:
        master_com = pickle.load(FILE)

    print("Committee size: {0}".format(len(master_com)))

    #if len(sys.argv) < 3:

    #columns = (2, -4, -3, -2, -1)
    #else:
    #    columns = [int(c) for c in sys.argv[2:]]
    P, T = parse_file(filename, targetcols = targets, inputcols = columns, normalize = True, separator = '\t', 
                      use_header = headers)

    #Need double brackets for dimensions to be right for numpy
    outputs = numpy.array([[master_com.risk_eval(inputs)] for inputs in P])
    c_index = get_C_index(T, outputs)

    print("C-Index: {0}".format(c_index))

    #if len(sys.argv) > 2:
    #    thresholds = [float(t) for t in sys.argv[2:]]
    #else:
    thresholds = None

    th = kaplanmeier(time_array = T[:, 0], event_array = T[:, 1], output_array = outputs, threshold = thresholds,
                     show_plot = False)
    #print("Threshold dividing the set in two equal pieces: " + str(th))
    if plt:
        plt.savefig('kaplanmeier_{0}_{1}.eps'.format(os.path.splitext(os.path.basename(savefile))[0], \
                                             os.path.splitext(os.path.basename(filename))[0]))
        
    #scatter(T[:, 0], outputs, T[:, 1], x_label = 'Target Data', y_label = 'Model Correlation',
    #        gridsize = 30, mincnt = 0, show_plot = False)
    #if plt:
    #    plt.savefig('scatter_cens_{0}_{1}.svg'.format(os.path.splitext(os.path.basename(savefile))[0], \
     #                                        os.path.splitext(os.path.basename(filename))[0]))
        #plt.show()

    output_file = '.test_{0}_{1}.cvs'.format(os.path.splitext(os.path.basename(savefile))[0], \
                                                              os.path.splitext(os.path.basename(filename))[0])
    with open(output_file, 'w') as F:
        #print('Targets\tOutputs\tEvents:')
        F.write("Targets\tOutputs\tEvents\n")
        for t, o in zip(T, outputs):
            #print("{0}\t{1}\t{2}".format(t[0], o[0], t[1]))
            F.write("{0}\t{1}\t{2}\n".format(t[0], o[0], t[1]))
            
    return output_file

if __name__ == '__main__':
    from scatterplot import scatterplot_files
    import sys

    if len(sys.argv) < 6:
        sys.exit('Proper usage is: {0} modelfile datafile targetcolumn, eventcolumn inputcol1 inputcol2 ...'.format(sys.argv[0]))
    
    model_file = sys.argv[1]
    filename = sys.argv[2]
    
    model_output_file = test_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], *sys.argv[5:])
    scatterplot_files(model_output_file, 0, 2, model_output_file, 1)
    
    
    
