import torch
import argparse
from model import AgentNetwork
from interval_timing import IntervalTiming1D
import json
import os
from prettytable import PrettyTable
import random
import numpy as np

torch.set_default_dtype(torch.float64)

class REINFORCE:

    def __init__(self, env, model, optimizer, agent_config, ckpt_path, test_episodes_per_interval):

        self.env = env
        self.model = model
        self.gamma = agent_config['gamma']
        self.log_interval = agent_config['log_interval']
        self.entropy_coeff = agent_config['entropy_coeff']
        self.optimizer = optimizer
        self.checkpoint_path = ckpt_path
        self.performance_log = {
            'env_config': {
                'stimuli': None,
                'scale': self.env.scale,
                'timings': self.env.timings,
                'rewards': self.env.rewards,
                'seed_offset': self.env.seed_offset
            },
            'encoder_config': self.model.encoder_config,
            'core_config': self.model.core_config,
            'actor_config': self.model.actor_config,
            'episodes': [],
            'average_accuracies': [],
            'average_scores': []
        }
        self.test_episodes_per_interval = test_episodes_per_interval

    def train(self, training_episodes):
        scores = []
        for episode in range(training_episodes):
            s, _ = self.env.reset(seed=random.randint(0, 1000))
            #print(self.env.sampled_interval)
            # print("Environment sample interval {}".format(self.env.sampled_interval))
            hist = []
            ep_score = 0
            done = False
            if isinstance(self.model.init_h, tuple):
                h = (self.model.init_h[0].clone(), self.model.init_h[1].clone())
            else:
                h = self.model.init_h.clone()
            while not done:
                act_prob, _, _, h = self.model(torch.tensor(s[None, :]), h)  # output dim: (1, 2)
                dist = torch.distributions.Categorical(probs=act_prob)
                entropy = dist.entropy()    # dim: (batch,)
                act = dist.sample()[0]  # stochastic, output dim scaler
                act_prob_a = act_prob[:, act]   # maybe use epsilon-greedy
                next_s, r, termination, truncation, info = self.env.step(act.detach().cpu().numpy())
                #print(s, act_prob, act, act_prob_a, r)
                done = termination | truncation
                hist.append((s, act_prob_a, entropy, r))  # maybe do a batch dimension instead of a list to go over for gradient update
                ep_score += r
                s = next_s

                if done:
                    scores.append(ep_score)
                    R = 0
                    loss = 0
                    for s, act_prob_a, entropy, r in hist[::-1]:
                        R = r + (self.gamma * R)
                        loss = -torch.sum(act_prob_a.log() * R) - (self.entropy_coeff * entropy) + loss     # summing up the losses
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    #print("Episode: {} Loss: {}".format(episode, loss.item()))

            if (episode + 1) % self.log_interval == 0:
                print("Episode: {} avg score: {}".format(episode + 1, np.mean(scores[-self.log_interval:])))

    def test(self):
        avg_score_intervals = {}
        avg_accuracy_intervals = {}
        self.model.eval()
        with torch.no_grad():
            for interval in self.env.stimuli+[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115]:
                scores_interval = []
                accuracies_interval = []
                for i in range(self.test_episodes_per_interval):
                    s, _ = self.env.reset(seed=0)
                    self.env.sampled_interval = interval
                    self.env.timings['measurement'] = self.env.sampled_interval * self.env.scale
                    # form the trial observations
                    self.env.measurement_obs = [1.] * self.env.scale + [0.] * (self.env.timings['measurement'] - 1 * self.env.scale)
                    self.env.trial_obs = self.env.fixation_obs + self.env.measurement_obs + self.env.production_obs
                    self.env.lower_bound = self.env.timings['fixation'] + 2 * self.env.timings['measurement'] - self.env.timings[
                        'measurement'] * self.env.tolerance
                    self.env.upper_bound = self.env.timings['fixation'] + 2 * self.env.timings['measurement'] + self.env.timings[
                        'measurement'] * self.env.tolerance

                    self.env.t = 0
                    self.env.info = {
                        'trial_interval': self.env.sampled_interval,
                        'period': 'fixation',
                        'gt': self.env.gt,
                        'current_t': self.env.t
                    }

                    # print("Environment sample interval {}".format(self.env.sampled_interval))
                    hist = []
                    ep_score = 0
                    done = False
                    if isinstance(self.model.init_h, tuple):
                        h = (self.model.init_h[0].clone(), self.model.init_h[1].clone())
                    else:
                        h = self.model.init_h.clone()
                    while not done:
                        act_prob, _, _, h = self.model(torch.tensor(s[None, :]), h)  # output dim: (1, 2)
                        dist = torch.distributions.Categorical(probs=act_prob)
                        entropy = dist.entropy()  # dim: (batch,)
                        act = dist.sample()[0]  # stochastic, output dim scaler
                        act_prob_a = act_prob[:, act]  # maybe use epsilon-greedy
                        next_s, r, termination, truncation, info = self.env.step(act.detach().cpu().numpy())
                        # print(s, act_prob, act, act_prob_a, r)
                        done = termination | truncation
                        hist.append((s, act_prob_a, entropy,
                                     r))  # maybe do a batch dimension instead of a list to go over for gradient update
                        ep_score += r
                        s = next_s

                        if done:
                            scores_interval.append(ep_score)
                            if ep_score == 10:
                                accuracies_interval.append(1)
                            else:
                                accuracies_interval.append(0)

                avg_score_intervals[interval] = np.mean(scores_interval)
                avg_accuracy_intervals[interval] = sum(accuracies_interval)/len(accuracies_interval)
        self.model.train()

        return avg_accuracy_intervals, avg_score_intervals


def check_eq_wt(m1, m2):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def count_parameters(model):
    # obtained from: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def parse_argument():

    parser = argparse.ArgumentParser()
    # network args
    parser.add_argument('--encoder_present', action='store_true', default=False, help='whether to use encoder or not')
    parser.add_argument('--encoder', type=str, default='mlp', choices=['mlp', 'conv'], help='encoder type to use')
    parser.add_argument('--core', type=str, default='sith', choices=['rnn', 'lstm', 'sith'], help='core type to use')
    #parser.add_argument('--core_freeze', action='store_true', default=False, help='whether to freeze the core weights or not')
    # encoder args - mlp
    parser.add_argument('--mlp_arch', type=int, nargs='+', default=[64, 1], help='mlp architecture: nodes of fc layers for encoder')
    parser.add_argument('--mlp_activations', type=str, nargs='+', default=['relu', 'relu'], help='activations for mlp network of actor')
    # encoder args - conv
    parser.add_argument('--conv_channels', type=int, nargs='+', default=[32, 64, 64], help='convolutional network channels for encoder')
    parser.add_argument('--conv_kernels', type=int, nargs='+', default=[8, 4, 3], help='convolutional network kernels for encoder')
    parser.add_argument('--conv_strides', type=int, nargs='+', default=[4, 2, 1], help='convolutional network kernels for encoder')
    parser.add_argument('--conv_activations', type=str, nargs='+', default=['relu', 'relu', 'relu'], help='convolutional network activations for encoder')
    parser.add_argument('--conv_mlp_arch', type=int, nargs='+', default=[512], help='mlp network after convolutional networkfor encoder')
    parser.add_argument('--conv_mlp_activations', type=str, nargs='+', default=['relu'], help='activations for mlp network after convolutional networkfor encoder')
    # core args - rnn args
    parser.add_argument('--rnn_num_layers', type=int, default=1, help='number of hidden node layers for rnn')
    parser.add_argument('--rnn_hidden_size', type=int, default=128, help='number of hidden nodes for rnn')
    parser.add_argument('--rnn_batch_first', type=bool, default=True, help='bath_first or not for rnn')
    # core args - lstm args
    parser.add_argument('--lstm_num_layers', type=int, default=1, help='number of hidden node layers for lstm')
    parser.add_argument('--lstm_hidden_size', type=int, default=128, help='number of hidden nodes for lstm')
    parser.add_argument('--lstm_batch_first', type=bool, default=True, help='bath_first or not for lstm')
    # core args - sith args
    parser.add_argument('--sith_tstr_min', type=float, default=1, help="minimum value for tau star")
    parser.add_argument('--sith_tstr_max', type=float, default=1000, help="maximum value for tau star")
    parser.add_argument('--sith_n_taus', type=int, default=80, help="number of tau stars")
    parser.add_argument('--sith_k', type=int, default=8, help='k value for sith')
    parser.add_argument('--sith_g', type=int, default=1, help='g value for sith')
    parser.add_argument('--sith_dt', type=float, default=1, help='dt value for sith')
    parser.add_argument('--sith_batch_first', type=bool, default=True, help='bath_first or not for sith')
    #parser.add_argument('--sith_in_alpha', action='store_true', default=False, help='whether to use alpha modulation for sith')
    parser.add_argument('--sith_F', action='store_true', default=False, help='whether to use F for sith output')
    # fc layer for actor
    parser.add_argument('--actor_type', type=str, default='mlp', choices=['mlp', 'conv'], help='what architecture to use for actor')
    parser.add_argument('--actor_arch', type=int, nargs='+', default=[64, 64], help='mlp architecture for actor')
    parser.add_argument('--actor_activations', type=str, nargs='+', default=['relu', 'relu'], help='activations for mlp network of actor')
    parser.add_argument('--act_conv_in_channel', type=int, default=1, help='input channel size for convolution network of actor')
    parser.add_argument('--act_conv_channels', type=int, nargs='+', default=[50], help='convolutional network channels for actor')
    parser.add_argument('--act_conv_kernels', type=int, nargs='+', default=[31], help='convolutional network kernels for actor')
    parser.add_argument('--act_conv_paddings', type=int, nargs='+', default=[20], help='convolutional network paddings for actor')
    parser.add_argument('--act_conv_strides', type=int, nargs='+', default=[1], help='convolutional network strides for actor')
    parser.add_argument('--act_conv_activations', type=str, nargs='+', default=['relu'], help='convolutional network activations for actor')
    parser.add_argument('--act_max_pool_kernel', type=int, default=50, help='max pool kernel size for actor')
    parser.add_argument('--act_mlp_net_arch', type=int, nargs='+', default=[50, 50], help='mlp network after convolutional network for actor')
    parser.add_argument('--act_mlp_activations', type=str, nargs='+', default=['relu', 'relu'], help='activations for mlp network after convolutional network for actor')

    # agent args
    parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
    parser.add_argument('--entropy_coeff', type=float, default=0.0001, help='weight on entropy loss')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--training_episodes', type=int, default=60000, help='episode to run for training')
    parser.add_argument('--log_interval', type=int, default=500, help='training steps before performing a test')
    # miscellaneous args
    parser.add_argument('--model_fname', type=str, default='test_interval_timing', help='file name of model to be saved')
    parser.add_argument('--model_seed', type=int, default=0, help='model seed')
    parser.add_argument('--env_scale', type=int, default=1, help='environment scale')
    arguments = parser.parse_args()

    return arguments


def main(args):
    with open("configs_fixed.json", 'r') as json_file:
        parsed_json = json_file.read()
    config_json = json.loads(parsed_json)  # for stationary parameters throughout the experiment
    encoder_config = None
    core_config = None

    if args.encoder == 'mlp':
        encoder_config = {
            'present': args.encoder_present,
            'type': 'mlp',
            'config': {
                'net_arch': args.mlp_arch,
                'activations': args.mlp_activations
            }
        }

    elif args.encoder == 'conv':
        encoder_config = {
            'type': 'conv',
            'conv_config': {
                'net_channels': args.conv_channels,
                'kernels': args.conv_kernels,
                'strides': args.conv_strides,
                'activations': args.conv_activations,
            },
            'mlp_config': {
                'net_arch': args.conv_mlp_arch,
                'activations': args.conv_mlp_activations
            }
        }

    if args.core == 'rnn':
        core_config = {
            "type": 'rnn',
            "num_layers": args.rnn_num_layers,
            "hidden_size": args.rnn_hidden_size,
            "batch_first": args.rnn_batch_first
        }
    elif args.core == 'lstm':
        core_config = {
            "type": 'lstm',
            "num_layers": args.lstm_num_layers,
            "hidden_size": args.lstm_hidden_size,
            "batch_first": args.lstm_batch_first
        }
    elif args.core == 'sith':
        core_config = {
            "type": 'sith',
            'tstr_min': args.sith_tstr_min,
            'tstr_max': args.sith_tstr_max,
            'n_taus': args.sith_n_taus,
            'k': args.sith_k,
            'dt': args.sith_dt,
            'g': args.sith_g,
            'batch_first': args.sith_batch_first,
            'F': args.sith_F
        }

    agent_config = {
        'gamma': args.gamma,
        'entropy_coeff': args.entropy_coeff/args.env_scale,
        'log_interval': args.log_interval
    }

    actor_config = {
        'type': args.actor_type,
        'net_arch': args.actor_arch,
        'activations': args.actor_activations,
        'conv': {
            'in_channel': args.act_conv_in_channel,
            'net_channels': args.act_conv_channels,
            'kernels': args.act_conv_kernels,
            'paddings': args.act_conv_paddings,
            'strides': args.act_conv_strides,
            'activations': args.act_conv_activations,
            'max_pool_kernel': args.act_max_pool_kernel,
            'mlp_net_arch': args.act_mlp_net_arch,
            'mlp_activations': args.act_mlp_activations
        }
    }

    env_configs = {
        'height': config_json['env_height'],
        'width': config_json['env_width'],
        'stimuli': config_json['env_stimuli'],  # [3018, 3310, 3629, 3979, 4363, 4784], [10, 12, 14, 16]
        'scale': args.env_scale,
        'timings': {
            'fixation': config_json['env_timings_fixation'],
            'delay': config_json['env_timings_delay'],
            'decision': config_json['env_timings_decision']
        },
        'rewards': {
            'correct': config_json['env_reward_correct'],
            'incorrect': config_json['env_reward_incorrect'],
        },
        'seed_offset': config_json['env_seed_offset']
    }

    env = IntervalTiming1D(**env_configs)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.n
    batch_size = 1
    torch.manual_seed(random.randint(0, 1000))  # args.model_seed
    model = AgentNetwork(obs_shape, act_shape, batch_size, encoder_config, core_config, actor_config)

    print(model)
    count_parameters(model)
    model_ckpt_folder = 'models/'
    os.makedirs(model_ckpt_folder, exist_ok=True)
    model_ckpt_filename = args.model_fname
    model_ckpt_path = model_ckpt_folder+model_ckpt_filename+'_model.pt'     # interval_timing_scale1_conv_latest
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    agent = REINFORCE(env, model, optimizer, agent_config, model_ckpt_path)
    agent.train(args.training_episodes)


if __name__ == "__main__":
    parsed_args = parse_argument()
    main(parsed_args)

