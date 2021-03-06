'''
Created on Aug 16, 2011

@author: jonask

Plots cross validation errors with error bars. The netsize on the bottom is the order of the files!!!
'''
import matplotlib
matplotlib.use('GTKAgg') #Only want to save images
import matplotlib.pyplot as plt
import sys
import numpy as np
from time import time
from math import sqrt

def reverse_error(e):
    ''' Reverse the error:
    1 / (C - 0.5) - 2
    1 / C
    '''
    #return 1.0 / (float(e) + 2.0) + 0.5
    return 1.0 / float(e)

def find_and_plot_winners(designs, *files):
    '''
    find_and_plot_winners(*files)

    Plots the winners of the model selection. Expects one or more filenames as input. The file are produced by
    model_finder.py
    Returns the highest rated design and saves the plot as designwinners_time
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

    fig = plt.figure()
    ax = fig.add_subplot(211)
    barax = fig.add_subplot(212, sharex = ax) #share x-axis
    ps = []
    labels = []

    trn = {}
    val = {}
    com_val = {}
    test = {}
    for filename in files:
        state = 'None'
        with open(filename) as FILE:

            for line in FILE:
                if line.startswith('Average Training Perf, Average Validation Perf, Average Committee Validation Perf, Test Perf, Design'):
                    state = 'result'
                    continue

                if state == 'result':
                    try:
                        vals = line.split(', ')
                        #print(vals)
                        design = str(vals[4]) + ', ' + str(vals[5])
                        #print design
                        if design not in trn:
                            trn[design] = []
                        if design not in val:
                            val[design] = []
                        if design not in com_val:
                            com_val[design] = []
                        if design not in test:
                            test[design] = []


                        trn[design].append(float(vals[0]))
                        val[design].append(float(vals[1]))
                        com_val[design].append(float(vals[2]))
                        test[design].append(float(vals[3]))
                    except IndexError:
                        continue

    trn_mean = {}
    val_mean = {}
    com_val_mean = {}
    test_mean = {}

    count = 0
    colors = ['k', 'r', 'b', 'g', 'y']
    maxcount = 0
    ticklabels = []
    done_arrow = True #Set to false to draw arrows and explanations
    try:
        all_designs = list(designs)
    except TypeError:
        #Not a list
        all_designs = []

    plotted_nodes = []

    best_winner = None

    trn_legend = None
    val_legend = None
    comval_legend = None
    test_legend = None

    for design in sorted(trn, key = lambda d: -len(test[d])): #Iterate based on the number of winners. Only relevant for the bar chart.
        #Since we are iterating in ascending order of number of winners, the last design will be the one we want to return
        winning_design = design
        design_tuple = winning_design.strip()
        design_tuple = (int(winning_design.split(',')[0][1:]), winning_design.split(',')[1][2:-2])
        nodes = design_tuple[0]
        plotted_nodes.append(nodes)
        nodes = str(nodes).strip()

        print winning_design
        count += 1
        ticklabels.append(nodes) #Remove trailing new line with strip

        trn_mean[design] = np.mean(trn[design])
        val_mean[design] = np.mean(val[design])
        com_val_mean[design] = np.mean(com_val[design])
        test_mean[design] = np.mean(test[design])


        plotlines, caplines, barlinecols = ax.errorbar(count - 0.1, trn_mean[design],
                                             yerr = [[trn_mean[design] - min(trn[design])], [-trn_mean[design] + max(trn[design])]],
                                             marker = 'o',
                                             color = 'k',
                                             ecolor = 'k',
                                             markerfacecolor = colors[count % len(colors)],
                                             label = design.strip() + ' trn',
                                             capsize = 5,
                                             linestyle = 'None')
        #ps.append(plotlines)
        trn_legend = plotlines

        #labels.append(design + ' trn')
        for entry in trn[design]:
            ax.plot(count - 0.1, entry, colors[count % len(colors)]+'+')

        if not done_arrow:
            texty = 20 if trn_mean[design] > val_mean[design] else -40
            ax.annotate('nettrn', xy=(count - 0.1, trn_mean[design]),
                    textcoords='offset points', xytext=(-30, texty),
                    arrowprops=dict(arrowstyle="->"), style='italic',
                    size='small')


        plotlines, caplines, barlinecols = ax.errorbar(count, val_mean[design],
                                             yerr = [[val_mean[design] - min(val[design])], [-val_mean[design] + max(val[design])]],
                                             marker = '^',
                                             color = 'k',
                                             ecolor = 'k',
                                             markerfacecolor = colors[count % len(colors)],
                                             label = design.strip() + ' val',
                                             capsize = 5,
                                             linestyle = 'None')
        #ps.append(plotlines)
        val_legend = plotlines

        #labels.append(design + ' val')
        for entry in val[design]:
            ax.plot(count, entry, colors[count % len(colors)]+'+')

        if not done_arrow:
            texty = 20 if trn_mean[design] < val_mean[design] else -40
            ax.annotate('netval', xy=(count, val_mean[design]),
            textcoords='offset points', xytext=(-40, -40),
            arrowprops=dict(arrowstyle="->"), style='italic',
            size='small')

        plotlines, caplines, barlinecols = ax.errorbar(count + 0.1, com_val_mean[design],
                                             yerr = [[com_val_mean[design] - min(com_val[design])], [-com_val_mean[design] + max(com_val[design])]],
                                             marker = 's',
                                             color = 'k',
                                             ecolor = 'k',
                                             markerfacecolor = colors[count % len(colors)],
                                             label = design.strip() + ' com_val',
                                             capsize = 5,
                                             linestyle = 'None')
        #ps.append(plotlines)
        comval_legend = plotlines

        #labels.append(design + ' val')
        for entry in com_val[design]:
            ax.plot(count + 0.1, entry, colors[count % len(colors)]+'+')

        if not done_arrow:
            texty = 20 if com_val_mean[design] > test_mean[design] else -40
            ax.annotate('comval', xy=(count+0.1, com_val_mean[design]),
                    textcoords='offset points', xytext=(-0, texty),
                    arrowprops=dict(arrowstyle="->"))

        plotlines, caplines, barlinecols = ax.errorbar(count + 0.2, test_mean[design],
                                             yerr = [[test_mean[design] - min(test[design])], [-test_mean[design] + max(test[design])]],
                                             marker = 'D',
                                             color = 'k',
                                             ecolor = 'k',
                                             markerfacecolor = colors[count % len(colors)],
                                             label = design.strip() + ' test',
                                             capsize = 5,
                                             linestyle = 'None')
        ps.append(plotlines)
        test_legend = plotlines

        labels.append(nodes)

        for entry in test[design]:
            ax.plot(count + 0.2, entry, colors[count % len(colors)]+'+')

        if not done_arrow:
            texty = 20 if com_val_mean[design] < test_mean[design] else -40
            ax.annotate('committee\ntest', xy=(count + 0.2, test_mean[design]),
                    textcoords='offset points', xytext=(-8, texty),
                    arrowprops=dict(arrowstyle="->"))
            done_arrow = True

        if len(test[design]) > maxcount:
            maxcount = len(test[design])
            best_winner = winning_design
        barax.bar(count - 0.2, len(test[design]), width=0.4, color=colors[count % len(colors)])
        barax.text(count, len(test[design]) + 0.1, str(len(test[design])), ha = 'center')

    try:
        for design in all_designs:
            nodes = design[0]
            if nodes not in plotted_nodes:
                count += 1
                #What ever is left should be plotted with zero
                labels.append(str(design[0]).strip())
                ticklabels.append(str(design[0]).strip())
                barax.bar(count - 0.2, 0, width=0.4, color=colors[count % len(colors)])
                barax.text(count, 0 + 0.1, "0", ha = 'center')
    except TypeError:
        #Not iterable list
        pass

    #leg = barax.legend(ps, labels, 'best')
    fig.legend([trn_legend, val_legend, comval_legend, test_legend],
              ['Training', 'Validation',
               'Ensamble validation', 'Ensamble test'],
              numpoints=1,

              loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)

              #bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    #ax.set_xlabel("Design (training, validation, committe validation, test) -->")
    ax.set_ylabel("Average C-Index")
    #ax.set_title('Network Design Cross validation C-Index results.')

    #plt.xlim(0, count + 1)
    #plt.ylim(0.5, 1.0)

    ax.set_ylim(ymin=0.5, ymax=1.0)
    ax.set_xlim(xmin=0.5, xmax=count+0.5)

    #Labels
    #lmin = lambda x: [min(y) for y in x.values()]
    #lmax = lambda x: [max(y) for y in x.values()]
    #ax.set_yticks(np.arange(min(lmin(trn) + lmin(val) + lmin(com_val) + lmin(test)), max(lmax(trn)) + 0.02, 0.02))

    ax.set_xticks(range(1, count + 1))
    ax.set_xticklabels(ticklabels)

    barax.set_ylim(ymin=0, ymax=maxcount*1.3)
    #Hide ticks
    #barax.get_yaxis().set_visible(False)
    barax.get_yaxis().set_ticks([])
    #barax.set_xlim(xmin=0.5, xmax=count+0.5)

    barax.set_ylabel("Winners")
    barax.set_xlabel("Hidden nodes")

    #barax.set_ticklabels(ticklabels)

    #plt.show()
    plt.tight_layout()
    fig.savefig('designwinners_{0:.0f}.eps'.format(time()))
    return best_winner

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: winner_extracter.py FILENAME1 FILENAME2 FILENAME3...')
    winning_design = find_and_plot_winners(None, *sys.argv[1:])
    print('Winning design = ' + winning_design)
