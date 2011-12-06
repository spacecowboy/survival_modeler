'''
Created on Sep 1, 2011

@author: jonask
'''

from kalderstam.util.filehandling import parse_file, get_cross_validation_sets
from survival.network import build_feedforward_committee
import numpy
from time import time
from survival.cox_error_in_c import get_C_index
from survival.cox_genetic import c_index_error
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

def model_contest(filename, columns, targets, designs):
    '''
    model_contest(filename, columns, targets, designs)
    
    You must use column names! Here are example values for the input arguments:
        
    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_the_n4369_dataset_with_logs_lymf.txt"
    columns = ('age', 'log(1+lymfmet)', 'n_pos', 'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos',
               'er_cyt_pos', 'size_gt_20', 'er_cyt_pos', 'pgr_cyt_pos')
    targets = ['time', 'event']
    
    Writes the results to '.winningdesigns_time.csv' and returns the filename
    '''
    
    starting_time = time.time()
    fastest_done = None
    m = Master()

    #m.connect('gibson.thep.lu.se', 'science')
    m.connect('130.235.189.249', 'science')
    print('Connected to server')
    m.clear_queues()

    print('\nIncluding columns: ' + str(columns))
    print('\nTarget columns: ' + str(targets))

    P, T = parse_file(filename, targetcols = targets, inputcols = columns, normalize = False, separator = '\t',
                      use_header = True)

    print("\nData set:")
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))
    print("T:" + str(T.shape))
    print("P:" + str(P.shape))

    comsize = 15 #Make sure it is divisible by three
    print('\nNumber of members in each committee: ' + str(comsize))

    print('Designs used in testing (size, function): ' + str(designs))

    val_pieces = 1
    print('Cross-test pieces: ' + str(val_pieces))

    cross_times = 12
    print('Number of times to repeat procedure: ' + str(cross_times))

    #try:
    #    pop_size = input('Population size [50]: ')
    #except SyntaxError as e:
    pop_size = 100
    print("Population size: " + str(pop_size))

    #try:
    #    mutation_rate = input('Please input a mutation rate (0.25): ')
    #except SyntaxError as e:
    mutation_rate = 0.25
    print("Mutation rate: " + str(mutation_rate))

    #try:
    #    epochs = input("Number of generations (200): ")
    #except SyntaxError as e:
    epochs = 100
    print("Epochs: " + str(epochs))

    print('\n Job status:\n')

    count = 0
    all_counts = []
    all_jobs = {}

    tests = {}
    all_best = []
    all_best_com_val = []
    all_best_avg_trn = []
    all_best_avg_val = []
    all_best_design = []
    all_best_test = []

    #Lambda times
    for _time in xrange(cross_times):
        #Get an independant test set, 1/tau of the total.
        super_set = get_cross_validation_sets(P, T, val_pieces , binary_column = 1)

        all_best.append({})
        all_best_com_val.append({})
        all_best_avg_trn.append({})
        all_best_avg_val.append({})
        all_best_design.append({})
        all_best_test.append({})

        best = all_best[_time]
        best_com_val = all_best_com_val[_time]
        best_avg_trn = all_best_avg_trn[_time]
        best_avg_val = all_best_avg_val[_time]
        best_design = all_best_design[_time]
        best_test = all_best_test[_time]


        #For every blind test group
        for ((TRN, TEST), _t) in zip(super_set, xrange(len(super_set))):
            TRN_INPUTS = TRN[0]
            TRN_TARGETS = TRN[1]
            TEST_INPUTS = TEST[0]
            TEST_TARGETS = TEST[1]

            #This for loop is not required, dealt with by running each architecture design on a separate machine
            best[_t] = None
            best_com_val[_t] = 0
            best_avg_trn[_t] = 0
            best_avg_val[_t] = 0
            best_design[_t] = None
            best_test[_t] = None

            for design in designs:
                count += 1
                all_counts.append(count)

                (netsize, hidden_func) = design

                com = build_feedforward_committee(comsize, len(P[0]), netsize, 1, hidden_function = hidden_func,
                                                  output_function = 'linear')

                tests[count] = (TEST_INPUTS, TEST_TARGETS)

                #1 is the column in the target array which holds the binary censoring information

                job = m.assemblejob((count, _time, _t, design),
                        train_committee, com, train_evolutionary, TRN_INPUTS,
                        TRN_TARGETS, binary_target = 1, epochs = epochs,
                        error_function = c_index_error, population_size =
                        pop_size, mutation_chance = mutation_rate)

                all_jobs[count] = job

                m.sendjob(job[0], job[1], *job[2], **job[3])

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
                print('Timed out after {0} seconds. Putting remaining jobs {1} back on the queue.\n \
                You should restart the server after this session.'.format(fastest_done, all_counts))
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

        TEST_INPUTS, TEST_TARGETS = tests[_c]

        all_counts.remove(_c)

        com.set_training_sets([set[0][0] for set in internal_sets]) #first 0 gives training sets, second 0 gives inputs.

        #Now what we'd like to do is get the value for each patient in the
        #validation set, for all validation sets. Then I'd like to average the
        #result for each such patient, over the different validation sets.
        allpats = []
        allpats.extend(internal_sets[0][0][0]) #Extend with training inputs
        allpats.extend(internal_sets[0][1][0]) #Extend with validation inputs

        allpats_targets = []
        allpats_targets.extend(internal_sets[0][0][1]) #training targets
        allpats_targets.extend(internal_sets[0][1][1]) #validation targets
        allpats_targets = numpy.array(allpats_targets)

        patvals = [[] for bah in xrange(len(allpats))]

        for pat, i in zip(allpats, xrange(len(patvals))):
            #We could speed this up by only reading every third dataset, but I'm not sure if they are ordered correctly
            for ((trn_in, trn_tar), (val_in, val_tar)), net in zip(internal_sets, com.nets):
                for valpat in val_in:
                    #Checks each variable individually, all() does a boolean and between the results
                    if (pat == valpat).all():
                        patvals[i].append(com.risk_eval(pat, net = net)) #Just to have something to count
                        break #Done with this data_set

        #Need  double brackets for dimensions to fit C-module
        avg_vals = numpy.array([[numpy.mean(patval)] for patval in patvals]) 
        #Now we have average validation ranks. do C-index on this
        avg_val_c_index = get_C_index(allpats_targets, avg_vals)

        trn_errors = numpy.array(trn_errors.values(), dtype = numpy.float64) ** -1
        vald_errors = numpy.array(vald_errors.values(), dtype = numpy.float64) ** -1
        avg_trn = numpy.mean(trn_errors)
        avg_val = numpy.mean(vald_errors)

        best = all_best[_time]
        best_com_val = all_best_com_val[_time]
        best_avg_trn = all_best_avg_trn[_time]
        best_avg_val = all_best_avg_val[_time]
        best_design = all_best_design[_time]
        best_test = all_best_test[_time]

        if avg_val_c_index > best_com_val[_t]:
            best[_t] = com
            best_com_val[_t] = avg_val_c_index
            best_avg_trn[_t] = avg_trn
            best_avg_val[_t] = avg_val
            best_design[_t] = design
            best_test[_t] = tests[_c]


    print('\nWinning designs')
    winnerfilename = '.winningdesigns_{0:.0f}.csv'.format(time())
    with open(winnerfilename, 'w') as F:
        print('Average Training Perf, Average Validation Perf, Average Committee Validation Perf, Test Perf, Design:')
        F.write('Average Training Perf, Average Validation Perf, Average Committee Validation Perf, Test Perf, Design\n')
        for _time in xrange(len(all_best)):
            best = all_best[_time]
            best_com_val = all_best_com_val[_time]
            best_avg_trn = all_best_avg_trn[_time]
            best_avg_val = all_best_avg_val[_time]
            best_design = all_best_design[_time]
            best_test = all_best_test[_time]
            for _t in best.keys():
                TEST_INPUTS, TEST_TARGETS = best_test[_t]
                com = best[_t]
    
                if len(TEST_INPUTS) > 0:
                    #Need double brackets for dimensions to be right for numpy
                    outputs = numpy.array([[com.risk_eval(inputs)] for inputs in TEST_INPUTS])
                    test_c_index = get_C_index(TEST_TARGETS, outputs)
                else:
                    test_c_index = 0
    
                print('{trn}, {val}, {com_val}, {test}, {dsn}'.format(trn = best_avg_trn[_t], val = best_avg_val[_t],
                      com_val = best_com_val[_t], test = test_c_index, dsn = best_design[_t]))
                F.write('{trn}, {val}, {com_val}, {test}, {dsn}\n'.format(trn = best_avg_trn[_t], val = best_avg_val[_t],
                        com_val = best_com_val[_t], test = test_c_index, dsn = best_design[_t]))
                        
    return winnerfilename

if __name__ == '__main__':
    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_the_n4369_dataset_with_logs_lymf.txt"
    columns = ('age', 'log(1+lymfmet)', 'n_pos', 'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos', 'er_cyt_pos', 'size_gt_20', 'er_cyt_pos', 'pgr_cyt_pos')
    targets = ['time', 'event']
    
    designs = [(1, 'linear')]
    [designs.append((i, 'tanh')) for i in [2, 3, 4, 6, 8, 10, 12, 15, 20]]
    
    model_contest(filename, columns, targets, designs)
