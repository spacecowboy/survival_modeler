import sys, pickle
from survival.cox_error_in_c import get_C_index
from kalderstam.util.filehandling import parse_file
from survival.plotting import kaplanmeier
import numpy

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None #This makes matplotlib optional

if __name__ == '__main__':
    savefile = "/home/gibson/jonask/Projects/Experiments/src/COXLARGE_COM.pcom"
    headers = False

    if len(sys.argv) < 2:
        filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"
        columns = (2, -5, -4, -3, -2, -1)
        targets = [4, 5]
    else:
        filename = sys.argv[1]
        targets = sys.argv[2:4] #2 and 3
        columns = tuple(sys.argv[4:]) #the rest
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

    print("targets: " + str(targets))
    print("inputs: " + str(columns))

    #if len(sys.argv) < 3:

    #columns = (2, -4, -3, -2, -1)
    #else:
    #    columns = [int(c) for c in sys.argv[2:]]
    P, T = parse_file(filename, targetcols = targets, inputcols = columns, normalize = False, separator = '\t', use_header = headers)

    outputs = numpy.array([[master_com.risk_eval(inputs)] for inputs in P]) #Need double brackets for dimensions to be right for numpy
    c_index = get_C_index(T, outputs)

    print("C-Index: {0}".format(c_index))

    #if len(sys.argv) > 2:
    #    thresholds = [float(t) for t in sys.argv[2:]]
    #else:
    thresholds = None

    th = kaplanmeier(time_array = T[:, 0], event_array = T[:, 1], output_array = outputs, threshold = thresholds)
    #print("Threshold dividing the set in two equal pieces: " + str(th))

    if plt:
        plt.show()

    print('Targets\tOutputs\tEvents:')
    for t, o in zip(T, outputs):
        print("{0}\t{1}\t{2}".format(t[0], o[0], t[1]))
