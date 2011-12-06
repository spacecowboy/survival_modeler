'''
Created on Sep 1, 2011

@author: jonask
'''

from kalderstam.util.filehandling import parse_file, get_cross_validation_sets
from survival.network import build_feedforward_committee
import numpy
from survival.cox_error_in_c import get_C_index
from survival.cox_genetic import c_index_error, weighted_c_index_error
import logging
from kalderstam.neural.training.davis_genetic import train_evolutionary
from Jobserver.master import Master
#from kalderstam.neural.training.genetic import train_evolutionary

try:
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    from survival.plotting import kaplanmeier
except ImportError:
    plt = None
except RuntimeError:
    plt = None

from kalderstam.neural.training.committee import train_committee
import sys
import time
import pickle

def com_cross():
    starting_time = time.time()
    fastest_done = None
    m = Master()

    #m.connect('gibson.thep.lu.se', 'science')
    m.connect('130.235.189.249', 'science')
    print('Connected to server')
    m.clear_queues()
    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_the_n4369_dataset_with_logs_lymf.txt"
    savefile = "/home/gibson/jonask/Projects/Experiments/src/COXLARGE_COM.pcom"

    columns = ('age', 'log(1+lymfmet)', 'n_pos', 'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos', 'er_cyt_pos', 'size_gt_20', 'er_cyt_pos', 'pgr_cyt_pos')
    targets = ['time', 'event']

    print('\nIncluding columns: ' + str(columns))
    print('Target columns: ' + str(targets))

    P, T = parse_file(filename, targetcols = targets, inputcols = columns, normalize = False, separator = '\t', use_header = True)

    #columns = (2, -6, -5, -4, -3, -2, -1)
    #_P, T = parse_file(filename, targetcols = [4, 5], inputcols = (2, -4, -3, -2, -1), ignorerows = [0], normalize = True)
    #P, _T = parse_file(filename, targetcols = [4], inputcols = columns, ignorerows = [0], normalize = True)

    print("\nData set:")
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))

    comsize = 3 * 20 #Make sure it is divisible by three (3*X will create X jobs)
    print('Number of members in the committee: ' + str(comsize))

    design = (4, 'tanh')
    print('Design used (size, function): ' + str(design))

    #try:
    #    pop_size = input('Population size [50]: ')
    #except SyntaxError as e:
    pop_size = 200
    print("Population size: " + str(pop_size))

    #try:
    #    mutation_rate = input('Please input a mutation rate (0.25): ')
    #except SyntaxError as e:
    mutation_rate = 0.25
    print("Mutation rate: " + str(mutation_rate))

    #try:
    #    epochs = input("Number of generations (200): ")
    #except SyntaxError as e:
    epochs = 200
    print("Epochs: " + str(epochs))

    #errorfunc = weighted_c_index_error
    errorfunc = c_index_error

    print("\nError function: " + errorfunc.__name__)

    print('\n Job status:\n')

    count = 0
    all_counts = []
    all_jobs = {}

    master_com = None

    allpats = P.copy()
    #allpats[:, 1] = 1 #This is the event column

    allpats_targets = T

    patvals = [[] for bah in xrange(len(allpats))]

    #Lambda times
    for _time in xrange(1):
        #Get an independant test set, 1/tau of the total.
        super_set = get_cross_validation_sets(P, T, 1, binary_column = 1)

        #For every blind test group
        for ((TRN, TEST), _t) in zip(super_set, xrange(len(super_set))):
            TRN_INPUTS = TRN[0]
            TRN_TARGETS = TRN[1]
            #TEST_INPUTS = TEST[0]
            #TEST_TARGETS = TEST[1]

            for com_num in xrange(comsize / 3):

                count += 1
                all_counts.append(count)

                (netsize, hidden_func) = design

                com = build_feedforward_committee(3, len(P[0]), netsize, 1, hidden_function = hidden_func, output_function = 'linear')

                #1 is the column in the target array which holds the binary censoring information

                job = m.assemblejob((count, _time, _t, design),
                    train_committee, com, train_evolutionary, TRN_INPUTS,
                    TRN_TARGETS, binary_target = 1, epochs = epochs,
                    error_function = errorfunc, population_size =
                    pop_size, mutation_chance = mutation_rate)

                all_jobs[count] = job

                m.sendjob(job[0], job[1], *job[2], **job[3])

#TIME TO RECEIVE THE RESULTS
    while(count > 0):
        print('Remaining jobs: {0}'.format(all_counts))
        if fastest_done is None:
            ID, RESULT = m.getresult() #Blocks
            fastest_done = time.time() - starting_time
        else:
            RETURNVALUE = m.get_waiting_result(2 * fastest_done)
            if RETURNVALUE is not None:
                ID, RESULT = RETURNVALUE
            else:
                print('Timed out after {0} seconds. Putting remaining jobs {1} back on the queue.\nYou should restart the server after this session.'.format(fastest_done, all_counts))
                for _c in all_counts:
                    job = all_jobs[_c]
                    m.sendjob(job[0], job[1], *job[2], **job[3])
                continue #Jump to next iteration

        print('Result received! Processing...')
        _c, _time, _t, design = ID

        (com, trn_errors, vald_errors, internal_sets) = RESULT

        if _c not in all_counts:
            print('This result [{0}] has already been processed.'.format(_c))
            continue

        count -= 1

        all_counts.remove(_c)

        com.set_training_sets([set[0][0] for set in internal_sets]) #first 0 gives training sets, second 0 gives inputs.

        if master_com is None:
            master_com = com
        else:
            master_com.nets.extend(com.nets) #Add this batch of networks

        #Now what we'd like to do is get the value for each patient in the
        #validation set, for all validation sets. Then I'd like to average the
        #result for each such patient, over the different validation sets.
        for pat, i in zip(allpats, xrange(len(patvals))):
            #We could speed this up by only reading every third dataset, but I'm not sure if they are ordered correctly...
            for ((trn_in, trn_tar), (val_in, val_tar)), net in zip(internal_sets, com.nets):
                for valpat in val_in:
                    if (pat == valpat).all(): #Checks each variable individually, all() does a boolean and between the results
                        patvals[i].append(com.risk_eval(pat, net = net)) #Just to have something to count
                        break #Done with this data_set

        avg_vals = numpy.array([[numpy.mean(patval)] for patval in patvals]) #Need  double brackets for dimensions to fit C-module
        #Now we have average validation ranks. do C-index on this
        avg_val_c_index = get_C_index(allpats_targets, avg_vals)
        print('Average validation C-Index so far: {0}'.format(avg_val_c_index))
        print('Saving committee so far in {0}'.format(savefile))
        with open(savefile, 'w') as FILE:
            pickle.dump(master_com, FILE)

if __name__ == '__main__':
    com_cross()
