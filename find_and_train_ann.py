# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:48:00 2011

@author: jonask
"""
from ann.model_finder import model_contest
from winner_extracter import find_and_plot_winners
from ann.model_trainer import train_model
from ann.model_tester import test_model

if __name__ == '__main__':
#    if len(sys.argv) < 5:
#        print('Proper usage is: {0} datafile, inputcolumns, targetcolumn, eventcolumn)'.format(sys.argv[0])
#        sys.exit
    
    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_the_n4369_dataset_with_logs_lymf.txt"
    columns = ('age', 'log(1+lymfmet)', 'n_pos', 'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos', 
               'er_cyt_pos', 'size_gt_20', 'er_cyt_pos', 'pgr_cyt_pos')
    targets = ['time', 'event']
    
    designs = [(1, 'linear')]
    [designs.append((i, 'tanh')) for i in [2, 3, 4, 6, 8, 10, 12, 15, 20]]
    
    print("\nSearching for the best model in " + str(designs))
    
    winnersfile = model_contest(filename, columns, targets, designs)
    
    print("\nWinners are stored in {0}, plotting contest...".format(winnersfile))
    
    winning_design = find_and_plot_winners(winnersfile)
    #Convert to Tuple
    winning_design = winning_design.strip()
    winning_design = (int(winning_design.split(',')[0][1:]), winning_design.split(',')[1][2:-2])
    
    print("\nTraining the winning design {0}...".format(str(winning_design)))
    
    model_file = train_model(winning_design, filename, columns, targets)
    
    print("\nProceeding with plotting on training data...")
    
    model_output_file = test_model(model_file, filename, targets[0], targets[1], *columns)
    
    print("\nModel output stored in {0}".format(model_output_file))
    
    #Against the same file, just another column
    #scatterplot_files(model_output_file, 0, 2, model_output_file, 1)
    
    cmd = 'test_model.py "{model_file}" YOUR_TEST_FILE_HERE "{0}" "{1}"'.format(targets[0], targets[1], \
                                                                                model_file = model_file)
    selfcmd = 'test_model.py "{model_file}" "{testfile}" "{0}" "{1}"'.format(targets[0], targets[1], \
                                                                        model_file = model_file, testfile = filename)
    for col in columns:
        cmd += ' "{0}"'.format(col)
        selfcmd += ' "{0}"'.format(col)
    
    print('''
    Completed.
    If you wish to test your model against some test data, use this command:
        {0}
    To get scatter plots for test data, run this command:
        {1}'''.format(cmd, selfcmd))