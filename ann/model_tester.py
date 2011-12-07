import pickle, os.path
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
    Saves a Kaplan-Meir plot as kaplanmeier_savefile_filename.svg
    Saves a file of the model output as .savefile_test_filename.cvs and returns this filename of the structure:
        Targets\tOutputs\tEvents
    '''
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
    P, T = parse_file(filename, targetcols = targets, inputcols = columns, normalize = False, separator = '\t', 
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
        plt.savefig('kaplanmeier_{0}_{1}.svg'.format(os.path.splitext(os.path.basename(savefile))[0], \
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
    savefile = "/home/gibson/jonask/Projects/Experiments/src/COXLARGE_COM.pcom"
    filename = ""
    
    
    