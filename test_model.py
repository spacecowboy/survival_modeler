# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:18:36 2011

@author: jonask
"""

from ann.model_tester import test_model
from scatterplot import scatterplot_files

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Proper usage is: {0} modelfile datafile targetcolumn, eventcolumn inputcol1 inputcol2 ...'.format(sys.argv[0])
        sys.exit()
    
    model_file = sys.argv[1]
    filename = sys.argv[2]
    
    model_output_file = test_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], tuple(sys.argv[5:]))
    scatterplot_files(model_output_file, 0, 2, model_output_file, 1)