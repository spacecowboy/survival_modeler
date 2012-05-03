'''
Created on Sep 1, 2011

@author: jonask
'''
from __future__ import division
from ann.filehandling import parse_file, get_cross_validation_sets

import numpy as np
from survival.cox_error_in_c import get_C_index

from pysurvival.cox import committee

import time
import pickle

def train_model(filename, columns, targets, separator = '\t', comsize=1):
    '''
    train_model(design, filename, columns, targets)

    Given a design, will train a committee like that on the data specified. Will save the committee as
    '.design_time.pcom' where design is replaced by the design and time is replaced by a string of numbers from time()
    Returns this filename
    '''
    headers = []
    headers.extend(columns)
    headers.extend(targets) #Add targets to the end

    targetcol = targets[0]
    eventcol = targets[1]

    savefile = ".cox_{time:.0f}.pcom".format(time = time.time())

    print('\nIncluding columns: ' + str(columns))
    print('Target columns: ' + str(targets))

    P, T = parse_file(filename, targetcols = targets, inputcols = columns, normalize = False, separator = separator, use_header = True)

    #columns = (2, -6, -5, -4, -3, -2, -1)
    #_P, T = parse_file(filename, targetcols = [4, 5], inputcols = (2, -4, -3, -2, -1), ignorerows = [0], normalize = True)
    #P, _T = parse_file(filename, targetcols = [4], inputcols = columns, ignorerows = [0], normalize = True)

    print("\nData set:")
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))

    print('Number of members in the committee: ' + str(comsize))

    allpats = P.copy()
    #allpats[:, 1] = 1 #This is the event column

    allpats_targets = T

    patvals = [[] for bah in xrange(len(allpats))]

    cox_committee = None

    #Get an independant test set, 1/tau of the total.
    super_set = get_cross_validation_sets(P, T, 1, binary_column = 1)

    #For every blind test group
    for ((TRN, TEST), _t) in zip(super_set, xrange(len(super_set))):
        TRN_INPUTS = TRN[0]
        TRN_TARGETS = TRN[1]
        #TEST_INPUTS = TEST[0]
        #TEST_TARGETS = TEST[1]

        #Modulo expressions mean we can deal with any number of committees, not only multiples of three
        _res = 1 if comsize == 1 else 0
        for com_num in xrange(int(comsize / 3) + int((comsize % 3) / 2) + _res):
            #Every time in the loop, create new validations sets of size 1/3. 3 everytime
            _tmp_val_sets = get_cross_validation_sets(TRN_INPUTS, TRN_TARGETS, 3, binary_column = 1)
            val_sets = []
	    if int(comsize / 3) > 0:
                _max = 3
            else:
                _max = int((comsize % 3) / 2) * 2 + _res
	    for _tmp_val_set in _tmp_val_sets[:_max]:
                ((trn_in, trn_tar), (val_in, val_tar)) = _tmp_val_set
                #Add target columns to the end
                _trn = np.append(trn_in, trn_tar, axis = 1)
                _val = np.append(val_in, val_tar, axis = 1)
                val_sets.append((_trn, _val))

            #And create 3 cox models, one for each validation
            tmp_com = committee(val_sets, targetcol, eventcol, headers)
	    print("Adding this many members: " + str(len(tmp_com)))
            if cox_committee is None:
                cox_committee = tmp_com
            else:
                #Extend the big committee
                cox_committee.members.extend(tmp_com.members)


    #Now what we'd like to do is get the value for each patient in the
    #validation set, for all validation sets. Then I'd like to average the
    #result for each such patient, over the different validation sets.
    print("Validating cox committee, this might take a little while...")
    _count = 0
    if len(cox_committee) < 3:
        allpats_targets = np.empty((0, 2)) #All patients won't be in the target set in this case
    for pat, i in zip(allpats, xrange(len(patvals))):
        if _count % 50 == 0:
            print("{0} / {1}".format(_count, len(patvals)))
        _count += 1
        #We could speed this up by only reading every third dataset, but I'm not sure if they are ordered correctly...
        for cox in cox_committee.members:
            (_trn, _val) = cox.internal_set
            trn_in = _trn[:, :-2] #Last two columns are targets
            val_in = _val[:, :-2]
            val_tar = _val[:, -2:]
            for valpat, valtar in zip(val_in, val_tar):
                if (pat == valpat).all(): #Checks each variable individually, all() does a boolean and between the results
                    patvals[i].append(cox_committee.risk_eval(pat, cox = cox)) #Just to have something to count
                    if len(cox_committee) < 3:
                        allpats_targets = np.append(allpats_targets, [valtar], axis = 0)
                    #print cox_committee.risk_eval(pat, cox = cox)
                    break #Done with this data_set

    avg_vals = []
    for patval in patvals:
        if len(patval) > 0:
            avg_vals.append([np.mean(patval)])
    avg_vals = np.array(avg_vals)
    #avg_vals = np.array([[np.mean(patval)] for patval in patvals]) #Need  double brackets for dimensions to fit C-module
    #Now we have average validation ranks. do C-index on this
    avg_val_c_index = get_C_index(allpats_targets, avg_vals)
    print('Average validation C-Index: {0}'.format(avg_val_c_index))
    print('Saving committee in {0}'.format(savefile))
    with open(savefile, 'w') as FILE:
        pickle.dump(cox_committee, FILE)

    return savefile

if __name__ == '__main__':
    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_the_n4369_dataset_with_logs_lymf.txt"

    columns = ('age', 'log(1+lymfmet)', 'n_pos', 'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos',
               'er_cyt_pos', 'size_gt_20', 'er_cyt_pos', 'pgr_cyt_pos')
    targets = ['time', 'event']
    train_model(filename, columns, targets)
