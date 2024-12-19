import numpy as np
import pickle
import os
import scipy

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'

def plot_performances_all(f_name, core):
    '''
    plot from a dictionary of performances for individual cores
    :param fname:
    :return:
    '''
    # loading the data
    with open("postprocessing/data/" + f_name + ".pkl", 'rb') as fp:
        performance_data = pickle.load(fp)

    os.makedirs("postprocessing/plots/", exist_ok=True)

    fig, ax = plt.subplots()
    ax.spines[['right', 'top']].set_visible(False)

    for key in performance_data.keys():
        if key in [core]:
            global_steps = performance_data[key]['global_steps'][0]  # assuming all the runs had the same global steps
            # performance_data[key]['values'])
            print(len(performance_data[key]['values'][0]))
            print(performance_data[key]['values'])
            print(len(performance_data[key]['values']))
            for i in range(len(performance_data[key]['values'])):
                plt.plot(performance_data[key]['global_steps'][i], performance_data[key]['values'][i], color='k', alpha=0.3)
            data = [performance_data[key]['values'][i] for i in range(len(performance_data[key]['values']))]
            data = np.array(data)
            #print(data.shape)
            average = np.mean(data, axis=0)
            #std_dev = np.std(data, axis=0)
            std_dev = scipy.stats.sem(data, axis=0)
            print(len(global_steps), average.shape)
            plt.plot(global_steps, average, color='r')  # mean and std_dev across trials

        # control over font type and size
        plt.xlabel(r'Global Environment Steps')
        plt.ylabel(r'Reward Mean')
        plt.ylim(0.4, 1.0)
        plt.xlim(0, 7000000)
        #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "$0$" if x == 0 else f"${int(x / 1_000_000)}\\mathrm{{M}}$"))
        plt.tight_layout()
        plt.savefig('postprocessing/plots/'+f_name+'_'+core+'_test_all.png', dpi=400)


def plot_performances(f_name):
    '''
    plot from a dictionary of performances
    :param fname:
    :return:
    '''
    # loading the data
    with open("postprocessing/data/" + f_name + ".pkl", 'rb') as fp:
        performance_data = pickle.load(fp)

    # getting average and standard deviation of the performances and plotting
    labels = {
        'rnn': "RNN",
        'lstm': "LSTM",
        'cogrnn_f_tilda': "$\\tilde{f}$",
        'cogrnn_f_tilda_dt_10': "$\\tilde{f}$",
        'cogrnn_f_tilda_dt_100': "$\\tilde{f}$",
        'cogrnn_F': "F",
        'rnn_frozen': "RNN_{frozen}",
        'lstm_frozen': "LSTM_{frozen}"
    }
    #markers
    markers = {
        'rnn': '.',
        'lstm': '^',
        'cogrnn_f_tilda': 's',
        'cogrnn_f_tilda_dt_10': 's',
        'cogrnn_f_tilda_dt_100': 's',
        'cogrnn_F': 'v',
        'rnn_frozen': 'p',
        'lstm_frozen': 'h'
    }
    fig, ax = plt.subplots()
    ax.spines[['right', 'top']].set_visible(False)

    for key in performance_data.keys():
        global_steps = performance_data[key]['global_steps'][0]  # assuming all the runs had the same global steps
        # performance_data[key]['values'])
        data = [performance_data[key]['values'][i] for i in range(len(performance_data[key]['values']))]
        data = np.array(data)
        #print(data.shape)
        average = np.mean(data, axis=0)
        std_dev = scipy.stats.sem(data, axis=0)
        #print(len(global_steps), average.shape)
        ax.plot(global_steps, average, label=labels[key], marker=markers[key], markersize=4)  # mean and std_dev across trials
        ax.fill_between(global_steps, average - std_dev, average + std_dev, alpha=0.4)

        # control over font type and size
        plt.xlabel(r'Global Environment Steps')
        plt.ylabel(r'Reward Mean')
        plt.ylim(0.4, 1.0)
        plt.xlim(0, 7000000)
        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4, frameon=False)
        plt.tight_layout()
        plt.savefig('postprocessing/plots/'+f_name+'.png', dpi=400)


if __name__ == '__main__':

    f_name = 'performances_2D_dt_100'
    cores = ['rnn', 'lstm', 'cogrnn']
    for core in cores:
        plot_performances_all(f_name, core) # plot the individual agent performance plot, one core at a time
    plot_performances(f_name) # plot Reward mean plots for different cores

