import torch
from collections import OrderedDict
torch.set_default_dtype(torch.float64)

def mlp_encoder(input_size, net_arch):
    '''
    defines a mlp using torch.nn.Sequential
    :param input_size:
    :param net_arch: list of nodes where each index corresponds to a nn layer.
    return a mlp instance
    '''

    od = OrderedDict()
    net_arch.insert(0, input_size)
    for i, node in enumerate(net_arch[:-1]):
        od.update({"fc_" + str(i + 1): torch.nn.Linear(node, net_arch[i + 1])})
        od.update({"activation_" + str(i + 1): torch.nn.ReLU()})
    mlp = torch.nn.Sequential(od)

    return mlp


def conv_mlp_encoder(input_shape, in_channel, conv_net_config, mlp_net_config):
    '''

    :param input_shape:
    :param in_channel:
    :param conv_net_config:
    :param mlp_net_config:
    :return:
    '''
    conv_arch = conv_net_config['net_channels']
    conv_activation = conv_net_config['activations']
    kernel_sizes = conv_net_config['kernels']
    strides = conv_net_config['strides']

    fc_arch = mlp_net_config['net_arch']
    fc_activation = mlp_net_config['activations']

    assert len(conv_arch) == len(conv_activation), f"number channels {conv_arch} and activation {conv_activation} is expected to be equal for conv network"
    conv_arch.insert(0, in_channel)
    od = OrderedDict()
    act = None
    act_count = 0
    for i, (channel, kernel_size, stride) in enumerate(zip(conv_arch, kernel_sizes, strides)):
        act_count += 1
        od.update({'conv'+str(i+1): torch.nn.Conv2d(in_channels=channel, out_channels=conv_arch[i+1], kernel_size=kernel_size, stride=stride)})
        if conv_activation[i] == 'relu':
            act = torch.nn.ReLU()
        elif conv_activation[i] == 'tanh':
            act = torch.nn.Tanh()
        od.update({conv_activation[i] + str(act_count): act})
        # calculating the output height and width, for: padding 0, dilation 1, stride 1 and square kernel
        input_shape[0] = (input_shape[0] - 1 * (kernel_size - 1) - 1)//stride + 1   #input height for the next layer
        input_shape[1] = (input_shape[1] - 1 * (kernel_size - 1) - 1)//stride + 1   #input width for the next layer

    assert len(fc_arch) == len(fc_activation), f"number nodes {fc_arch} and activation {fc_activation} is expected to be equal for fc network"
    od.update({'flatten': torch.nn.Flatten(start_dim=1)})

    fc_arch.insert(0, conv_arch[-1]*input_shape[0]*input_shape[1])
    for i, node in enumerate(fc_arch[:-1]):
        act_count += 1
        od.update({'fc'+str(i+1): torch.nn.Linear(node, fc_arch[i+1])})
        if fc_activation[i] == 'relu':
            act = torch.nn.ReLU()
        elif fc_activation[i] == 'tanh':
            act = torch.nn.Tanh()
        od.update({fc_activation[i]+str(act_count): act})

    encoder = torch.nn.Sequential(od)

    return encoder


def build_critic(input_size, critic_config):

    critic_arch = critic_config['net_arch']
    critic_activation = critic_config['activations']

    critic_arch.insert(0, input_size)
    od = OrderedDict()
    act = None
    act_count = 0
    for i, node in enumerate(critic_arch[:-1]):
        act_count += 1
        od.update({'fc'+str(i+1): torch.nn.Linear(node, critic_arch[i+1])})
        if critic_activation[i] == 'relu':
            act = torch.nn.ReLU()
        elif critic_activation[i] == 'tanh':
            act = torch.nn.Tanh()
        od.update({critic_activation[i]+str(act_count): act})

    od.update({'output': torch.nn.Linear(critic_arch[-1], 1)})
    critic = torch.nn.Sequential(od)

    return critic


def build_actor(input_size, actor_config, num_actions):
    actor_arch = actor_config['net_arch']
    actor_activation = actor_config['activations']

    actor_arch.insert(0, input_size)
    od = OrderedDict()
    act = None
    act_count = 0
    for i, node in enumerate(actor_arch[:-1]):
        act_count += 1
        od.update({'fc' + str(i + 1): torch.nn.Linear(node, actor_arch[i + 1])})
        if actor_activation[i] == 'relu':
            act = torch.nn.ReLU()
        elif actor_activation[i] == 'tanh':
            act = torch.nn.Tanh()
        od.update({actor_activation[i] + str(act_count): act})

    od.update({'output': torch.nn.Linear(actor_arch[-1], num_actions)})
    actor = torch.nn.Sequential(od)

    return actor
