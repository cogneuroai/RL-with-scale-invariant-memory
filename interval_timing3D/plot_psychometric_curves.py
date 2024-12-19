import numpy as np
import os
import pickle, gzip
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 15
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'


def plot_psychometric_curves(f_name):
    '''

    :param f_name:
    :return:
    '''
    alpha = 0.1  # for confidence interval
    os.makedirs("postprocessing/plots/", exist_ok=True)

    # load the data
    logs = pickle.load(gzip.open("postprocessing/data/" + f_name + 'validation_test_logs.pkl.gz', 'r'))
    # creating a dictionary to hold a list information as lists
    test_info_dict_all = {
        'trial_stimulus': [],
        'trial_gt': [],
        'trial_reward': [],
        'trial_last_action': [],
        'trial_success': [],
    }

    for i in range(len(logs)):
        for j in range(len(logs[i])):
            for key in test_info_dict_all.keys():
                test_info_dict_all[key].append(logs[i][j][key])

    # psychometric curve
    trial_stims = np.array(test_info_dict_all['trial_stimulus'])
    stim_values = np.unique(test_info_dict_all['trial_stimulus'])  # unique stimulus
    print(stim_values)
    success = np.array(test_info_dict_all['trial_success'])

    results = {}
    # adding new keys to the results dictionary
    for u in stim_values:
        results[str(u)] = []

    mid_point = stim_values.shape[0] // 2
    #long = stim_values[mid_point:].copy()
    short = stim_values[:mid_point].copy()

    for i in range(trial_stims.shape[0]):
        if trial_stims[i] in short:
            results[str(trial_stims[i])].append(1 - int(success[i]))
        else:
            results[str(trial_stims[i])].append(int(success[i]))

    # for storing the probabilities to choose interval choice long given a stimulus
    plong = {}
    for key in results.keys():
        plong[key] = {'mean': sum(results[key]) / len(results[key]),
                      'range': proportion_confint(sum(results[key]), len(results[key]), alpha, "jeffrey")}

    # obtain the y, yerr to plot the psychometric curve
    y = [plong[key]['mean'] for key in plong.keys()]

    fig, ax = plt.subplots()
    ax.spines[['right', 'top']].set_visible(False)

    yerr = [plong[key]['range'][1] - plong[key]['range'][0] for key in plong.keys()]

    ax.errorbar([r'$3018$', r'$3310$', r'$3629$', r'$3929$', r'$4364$', r'$4764$'], y, yerr, fmt='o-', color='blue', ecolor='red', elinewidth=3, capsize=0)

    #plt.scatter(x, y)
    #plt.vlines(x, ymin=lower, ymax=upper)
    plt.xlabel('Stimulus in milliseconds')
    plt.ylabel('P_long')
    plt.gca().set_ylim(top=1.0)
    plt.tight_layout()
    plt.savefig('postprocessing/plots/' + f_name + '_psychometric_curve_diff_interval.png', dpi=400)
    plt.clf()

if __name__ == '__main__':

    f_names = ['cogrnn', 'lstm', 'rnn']
    for f_name in f_names:
        plot_psychometric_curves(f_name)


