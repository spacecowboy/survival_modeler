import numpy
import matplotlib.pyplot as plt
import sys
from survival.plotting import scatter
from survival.cox_error_in_c import get_C_index

if len(sys.argv) < 3:
    sys.exit('Not enough arguments. Takes two files!')

if len(sys.argv) >= 4:
    first_col = sys.argv[3]
else:
    first_col = 0

if len(sys.argv) >= 5:
    second_col = sys.argv[4]
else:
    second_col = 1

if len(sys.argv) >= 6:
    event_col = sys.argv[5]
else:
    event_col = None

with open(sys.argv[1], 'r') as f:
    X_in = [line.split() for line in f.readlines()]
X_in = numpy.array(X_in)
X = X_in[1:, first_col]
X = numpy.array(X, dtype = 'float')

with open(sys.argv[2], 'r') as f:
    Y_in = [line.split() for line in f.readlines()]

Y_in = numpy.array(Y_in)
Y = Y_in[1:, second_col]
Y = numpy.array(Y, dtype = 'float')

if event_col is not None:
    events = X_in[1:, event_col]
    events = numpy.array(events, dtype = 'float')
    print 'Using events'
else:
    events = None
    
T = numpy.empty((len(X), 2), dtype='float')
T[:, 0] = X
T[:, 1] = events
outputs = numpy.empty((len(X), 2), dtype='float')
outputs[:, 0 ] = Y
outputs[:, 1] = events
c_index = get_C_index(T, outputs)
print("C-Index between these files is: {0}".format(c_index))

scatter(X.copy(), Y.copy(), events = events, gridsize = 30, mincnt = 0, show_plot = False)
plt.title("Scatter plot heatmat, taking censoring into account\nC-Index between these files is: {0}".format(c_index))
plt.xlabel(sys.argv[1])
plt.ylabel(sys.argv[2])

plt.figure()

#plt.figure(2)
plt.xlabel(sys.argv[1])
plt.ylabel(sys.argv[2])
#plt.scatter(X, Y, c = 'r', marker = '+')
#plt.plot(Y, outputs, 'gs')
#plt.plot(Y, outputs, 'b:')

import matplotlib.cm as cm
import numpy as np
x, y = X, Y
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

plt.subplot(111)
plt.hexbin(x, y, bins = 'log', cmap = cm.jet, gridsize = 30, mincnt = 0)
plt.axis([xmin, xmax, ymin, ymax])
plt.title("Scatter plot heatmat, logarithmic count\nC-Index between these files is: {0}".format(c_index))
cb = plt.colorbar()
cb.set_label('log10(N)')

plt.show()
