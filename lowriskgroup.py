#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage: program.py


'''
from __future__ import division
import sys
import pickle
import numpy as np
from ann.filehandling import parse_file, normalizeArrayLike
from survival.plotting import kaplanmeier
from annplot import show, save, plot_roc

def plotKM(targets, outputs, cut):
    kaplanmeier(time_array=targets[:,0], event_array=targets[:, 1],
                output_array=outputs, threshold=cut, show_plot=False)


def plotRoc(targets, outputs, labels=None, limit=None):
    '''Outputs is an array of several model outputs!'''
    if limit is None:
        limit = max(targets[:, 0])
    print("limit", limit)

    #Remove censored before limit!
    dead = targets[:, 1] > 0
    alive = targets[:, 0] >= limit

    for i, isdead in enumerate(dead):
        if isdead and targets[i, 0] > limit:
            dead[i] = False
        elif isdead and targets[i, 0] == limit:
            alive[i] = False

    print("all: ", (alive + dead).all())

    print("dead {}  alive {} total {}".format(len(targets[dead]),
                                              len(targets[alive]),
                                              len(targets)))

    #Specify our new 0 - 1 classes targets
    classes = np.zeros(len(targets))
    classes[dead] = 0
    classes[alive] = 1
    #Get rid of the censored in between
    classes = np.append(classes[dead], classes[alive])

    print("classes len:", len(classes))

    #Outputs must follow suite
    clean_outputs = []
    for output in outputs:
        clean_outputs.append(np.append(output[dead], output[alive]))

    plot_roc(classes, clean_outputs, labels)


def loadData():
    trnfile = ('/home/gibson/jonask/DataSets/breast_cancer_1/' +
               'n4369_trainingtwothirds.csv')
    testfile = ('/home/gibson/jonask/DataSets/breast_cancer_1/' +
                'n4369_targetthird.csv')
    columns = ['age', 'log(1+lymfmet)', 'n_pos', 'tumsize',
               'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos',
               'er_cyt_pos', 'size_gt_20', 'er_cyt', 'pgr_cyt']
    targets = ['time_10y', 'event_10y']

    # Normalize the test data as we normalized the training data
    normP, bah = parse_file(trnfile, inputcols=columns,
                            targetcols=targets, normalize=False,
                            separator=',', use_header=True)

    unNormedTestP, test_targets = parse_file(testfile, inputcols=columns,
                                  targetcols=targets, normalize=False,
                                  separator=',', use_header=True)

    test_data = normalizeArrayLike(unNormedTestP, normP)

    #If you want to return train data instead
    trn_data, trn_targets = parse_file(trnfile, inputcols=columns,
                            targetcols=targets, normalize=True,
                            separator=',', use_header=True)

    #return trn_data, trn_targets
    return test_data, test_targets

def getFalsePositives(targets, subset, limit=None):
    '''Find out how many elements are wrongfully categorized in the subset for
    the output.'''
    #Find what the ultimate censoring is
    if limit is None:
        limit = max(targets[:, 0])

    wrong = total = 0
    for idx in subset:
        #People alive at the limit are good
        if targets[idx, 0] >= limit:
            total += 1
        #Only un-censored can lower the score
        elif targets[idx, 1]:
            wrong += 1
            total += 1

    #and what fraction is that of the whole subset
    if not total:
        return 1
    else:
        score = wrong / total
        return score


def getLargestLowRisk(targets, sortidx, falsepositiverate=0.2):
    #Find the largest subset of long times (low risk) that gives at most 10%
    #false positives.
    final_score = 0
    largest_subset = subset = np.int_([])

    for idx in sortidx:
        #This creates a new list, so doesn't affect largest
        subset = np.append(subset, idx)
        #Get false positive rate for this subset
        score = getFalsePositives(targets, subset)

        if score <= falsepositiverate:
            final_score = score
            largest_subset = subset

    #Print some information about this
    print(("Size is {2:.2f} ({0}/{1}) with false positive"
           + " rate of {3:.2f}").format(len(largest_subset), len(targets),
                                    len(largest_subset) / len(targets),
                                    final_score))


    return largest_subset

if __name__ == "__main__":
    if len(sys.argv) > 1:
        FALSE_POS = float(sys.argv[1])
    else:
        FALSE_POS = 0.1

    indata, targets = loadData()

    with open(('/home/gibson/jonask/Dropbox/Ann-Survival-Phd/publication_data' +
               '/cox/single/10_year/cox_1328777311.pcom'), 'r') as FILE:
        cox = pickle.load(FILE)

    #outputs = cox.sim(indata)
    cox_outputs = np.array([[cox.risk_eval(inputs)] for inputs in indata])

    #Sort it according to time (reversed). Sort along the rows, but only use
    #information from time column
    coxidx = np.argsort(cox_outputs, axis=0)[::-1, 0]

    print("\nCox's best group...")
    cox_low_risk = getLargestLowRisk(targets, coxidx, FALSE_POS)

    #Evaluate ann on the same
    with open(('/home/gibson/jonask/Dropbox/Ann-Survival-Phd/publication_data' +
               '/ann/cens_10y/2_tanh_1328829716.pcom'), 'r') as FILE:
        com = pickle.load(FILE)

    #outputs = com.sim(indata)
    ann_outputs = np.array([[com.risk_eval(inputs)] for inputs in indata])

    annidx = np.argsort(ann_outputs, axis=0)[::-1, 0]

    #Get ann score on the same size subset
    ann_score = getFalsePositives(targets, annidx[:len(cox_low_risk)])

    #Print some info about it
    print("\nANN false positive rate on the same = {}".format(ann_score))

    #Now find what the largest group for the ann would have been, and cox's
    #result on the same

    print("\nANN's best group...")
    ann_low_risk = getLargestLowRisk(targets, annidx, FALSE_POS)

    cox_score = getFalsePositives(targets, coxidx[:len(ann_low_risk)])

    #Print some info about it
    print("\nCox false positive rate on the same = {}".format(cox_score))

    if len(cox_low_risk) > 0:
        ann_cut = min(ann_outputs[annidx[:len(cox_low_risk)], 0])
        cox_cut = min(cox_outputs[cox_low_risk, 0])

        plotKM(targets, ann_outputs, ann_cut)
        plotKM(targets, cox_outputs, cox_cut)

    plotRoc(targets, [ann_outputs, cox_outputs], ['ANN', 'COX'], limit=10)

    if len(ann_low_risk) > 0:
        ann_cut = min(ann_outputs[ann_low_risk, 0])
        cox_cut = min(cox_outputs[coxidx[:len(ann_low_risk)], 0])

        plotKM(targets, ann_outputs, ann_cut)
        plotKM(targets, cox_outputs, cox_cut)

    show()
