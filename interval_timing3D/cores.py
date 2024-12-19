import torch
from CogRNN import CogRNN
torch.set_default_dtype(torch.float64)


def lstm_core(input_size, net_config):
    '''
    defines a lstm_core
    :param input_size:
    :param net_config: input_size, hidden_size, num_layers, and batch_first
    :return: returns a lstm instance
    '''
    return torch.nn.LSTM(input_size=input_size, hidden_size=net_config['hidden_size'], num_layers=net_config['num_layers'], batch_first=net_config['batch_first'])

def rnn_core(input_size, net_config):
    '''
    defines a rnn core
    :param input_size:
    :param net_config: input_size, hidden_size, num_layers, and batch_first
    :return: returns a rnn instance
    '''
    return torch.nn.RNN(input_size=input_size, hidden_size=net_config['hidden_size'], num_layers=net_config['num_layers'], batch_first=net_config['batch_first'])

def cogrnn_core(config):
    '''
    defines a cogrnn core
    :param config: tstr_min, tstr_max, n_taus, k, g, dt, batch_first
    :return: a cogrnn instance
    '''
    return CogRNN(tstr_min=config['tstr_min'], tstr_max=config['tstr_max'], n_taus=config['n_taus'], k=config['k'], dt=config['dt'], g=config['g'], batch_first=config['batch_first'])
