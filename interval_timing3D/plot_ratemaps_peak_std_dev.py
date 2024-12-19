import numpy as np
import os
import pickle, gzip
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 15
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'


def gaussian(x, amplitude, mu, sigma):
    '''
    curve fitting with this gaussian
    '''
    return amplitude*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-mu)/sigma)**2)))


def plot_ratemaps_peak_std(f_name):
    '''
    plot neuron activations for different cores and peak vs standard deviation plots
    :param f_name:
    :return:
    '''
    os.makedirs("postprocessing/plots/", exist_ok=True)

    # load the data
    logs = pickle.load(gzip.open('postprocessing/data/' + f_name + 'validation_test_logs.pkl.gz', 'r'))
    # ratemaps
    # go through the test info list, store the mem_signals according to the stimulus value
    results = {}
    stim_values = [4784] # storing only the longest interval
    # adding new keys to the results dictionary
    for u in stim_values:
        results[str(u)] = []

    for i in range(len(logs)):
        for j in range(len(logs[i])):
            if logs[i][j]['trial_stimulus'] == stim_values[-1]:
                # largest stimuli storing across different agents
                results[str(logs[i][j]['trial_stimulus'])].append(logs[i][j]['mem_signals'])
                # breaking as each agent should have the same representation of the same interval
                break

    print(results.keys())
    for key in results.keys():
        print(len(results[key]))
        for i in range(len(results[key])):
            ratemap_hist = np.array(results[key][i]).copy()
            # print(ratemap_hist.shape)
            normalized_ratemap_hist = (ratemap_hist - np.min(ratemap_hist, axis=1).reshape(-1, 1)) / np.ptp(ratemap_hist,axis=1).reshape(-1, 1)
            normalized_ratemap_hist = np.where(np.isnan(normalized_ratemap_hist), 0, normalized_ratemap_hist)  # getting rid of nans
            normalized_ratemap_hist = np.where(np.isinf(normalized_ratemap_hist), 0, normalized_ratemap_hist)  # getting rid of +/- inf
            # sorting the rows using argsort
            id_max = np.argmax(normalized_ratemap_hist, axis=1)
            id_sorted = np.argsort(id_max)
            normalized_ratemap_hist_sorted = normalized_ratemap_hist[id_sorted]

            plt.imshow(normalized_ratemap_hist_sorted, aspect='auto')
            plt.xlabel('Time')  # may need to change the ticks for actual time
            plt.ylabel('Neurons')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('postprocessing/plots/' + f_name + '_agent_' + str(i) + '_largest_stim_ratemap_not_clean.png', dpi=400)
            plt.clf()

            # clean the neurons
            constant_idx = []
            for j in range(normalized_ratemap_hist_sorted.shape[0]):
                # not changing
                if ((not np.all(np.gradient(normalized_ratemap_hist_sorted[j]))) or (np.sum(normalized_ratemap_hist_sorted[j])==0.0) or (np.sum(normalized_ratemap_hist_sorted[j])>40)):
                    constant_idx.append(j)

            # check peak for the neurons
            peaks = np.argmax(normalized_ratemap_hist_sorted, axis=1)
            selected = set(range(normalized_ratemap_hist_sorted.shape[0])) - set(constant_idx) - set(np.where(peaks < 3)[0].tolist()) - set(np.where(peaks > 46)[0].tolist())

            selected_normalized_ratemap_hist_sorted = normalized_ratemap_hist_sorted[list(selected)].copy()

            plt.imshow(selected_normalized_ratemap_hist_sorted[:, 3:48], aspect='auto') # discarding the fixatio and decision period

            #plt.imshow(normalized_ratemap_hist_sorted, aspect='auto')
            plt.xlabel('Time')  # may need to change the ticks for actual time
            plt.ylabel('Neurons')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('postprocessing/plots/' + f_name + '_agent_'+str(i)+'_largest_stim_ratemap_clean.png', dpi=400)
            plt.clf()

            selected_ramping_decying = set(range(normalized_ratemap_hist_sorted.shape[0])) - selected - set(constant_idx)
            selected_ramping_decying_normalized_ratemap_hist = normalized_ratemap_hist_sorted[list(selected_ramping_decying)].copy()

            id_max = np.argmax(selected_ramping_decying_normalized_ratemap_hist, axis=1)
            id_sorted = np.argsort(id_max)
            selected_ramping_decying_normalized_ratemap_hist_sorted = selected_ramping_decying_normalized_ratemap_hist[id_sorted]
            ax = plt.figure().gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            plt.imshow(selected_ramping_decying_normalized_ratemap_hist_sorted[:, 3:48], aspect='auto')

            plt.xlabel('Time')  # may need to change the ticks for actual time
            plt.ylabel('Neurons')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('postprocessing/plots/' + f_name + '_agent_' + str(i) + '_largest_stim_ratemap_decay_ramp.png', dpi=400)
            plt.clf()

            # mode vs half width scatter plots
            # find the peaks of the neurons, and width at half height
            selected_peaks = np.argmax(selected_normalized_ratemap_hist_sorted, axis=1)

            std_devs = []
            mu_s = []
            selected = []
            for k in range(selected_normalized_ratemap_hist_sorted.shape[0]):
                print(k)
                try:
                    (a, mu, sigma), _ = curve_fit(gaussian, np.array(range(47)), selected_normalized_ratemap_hist_sorted[k, 3:50], p0=[100, 10, 200], maxfev=8000)
                    if sigma <= 25:
                        mu_s.append(mu)
                        std_devs.append(sigma)
                        selected.append(k)
                except:
                    print("Could not find any estimation of the parameters")

            plt.scatter(selected_peaks[selected], std_devs)
            plt.xlabel('Peaks')
            plt.ylabel('Standard Deviations')
            plt.gca().set_ylim(top=25.0)
            plt.tight_layout()
            plt.savefig('postprocessing/plots/' + f_name + '_agent_' + str(i) + '_largest_stim_mode_vs_std_dev.png', dpi=400)
            plt.clf()


if __name__ == '__main__':

    f_names = ['cogrnn', 'cogrnn_F', 'lstm', 'rnn']
    for f_name in f_names:
        plot_ratemaps_peak_std(f_name)

