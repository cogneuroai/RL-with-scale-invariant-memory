import gymnasium as gym
import torch
import argparse
from env import TempBisectionEnv3D, make_env3D, ScaleObs
from model import AgentNetwork
from agents import A2CAgent
import json
import os
from prettytable import PrettyTable

import wandb
import warnings


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
    parser.add_argument('--encoder', type=str, default='conv', choices=['mlp', 'conv'], help='encoder type to use')
    parser.add_argument('--core', type=str, default='lstm', choices=['rnn', 'lstm', 'cogrnn'], help='core type to use')
    parser.add_argument('--core_freeze', action='store_true', default=False,
                        help='whether to freeze the core weights or not')
    # encoder args - mlp
    parser.add_argument('--mlp_arch', type=int, nargs='+', default=[64, 64],
                        help='mlp architecture: nodes of fc layers for encoder')
    # encoder args - conv
    parser.add_argument('--conv_channels', type=int, nargs='+', default=[32, 16, 32],
                        help='convolutional network channels for encoder')
    parser.add_argument('--conv_kernels', type=int, nargs='+', default=[8, 4, 8],
                        help='convolutional network kernels for encoder')
    parser.add_argument('--conv_strides', type=int, nargs='+', default=[2, 1, 2],
                        help='convolutional network kernels for encoder')
    parser.add_argument('--conv_activations', type=str, nargs='+', default=['relu', 'relu', 'relu'],
                        help='convolutional network activations for encoder')
    parser.add_argument('--conv_mlp_arch', type=int, nargs='+', default=[64],
                        help='mlp network after convolutional networkfor encoder')
    parser.add_argument('--conv_mlp_activations', type=str, nargs='+', default=['relu'],
                        help='activations for mlp network after convolutional networkfor encoder')
    # core args - rnn args
    parser.add_argument('--rnn_num_layers', type=int, default=1, help='number of hidden node layers for rnn')
    parser.add_argument('--rnn_n_taus', type=int, default=20,
                        help='portions to take from a batch of rnn output as tau starts')
    parser.add_argument('--rnn_batch_first', type=bool, default=True, help='bath_first or not for rnn')
    # core args - lstm args
    parser.add_argument('--lstm_num_layers', type=int, default=1, help='number of hidden node layers for lstm')
    parser.add_argument('--lstm_n_taus', type=int, default=8,
                        help='portions to take from a batch of lstm output as tau starts')
    parser.add_argument('--lstm_batch_first', type=bool, default=True, help='bath_first or not for lstm')
    # core args
    parser.add_argument('--cogrnn_tstr_min', type=float, default=1, help="minimum value for tau star")
    parser.add_argument('--cogrnn_tstr_max', type=float, default=1000, help="maximum value for tau star")
    parser.add_argument('--cogrnn_n_taus', type=int, default=80, help="number of tau stars")
    parser.add_argument('--cogrnn_k', type=int, default=8, help='k value for cogrnn')
    parser.add_argument('--cogrnn_g', type=int, default=1, help='g value for cogrnn')
    parser.add_argument('--cogrnn_dt', type=float, default=1, help='dt value for cogrnn')
    parser.add_argument('--cogrnn_batch_first', type=bool, default=True, help='bath_first or not for cogrnn')
    parser.add_argument('--cogrnn_in_alpha', action='store_true', default=False,
                        help='whether to use alpha modulation for cogrnn')
    parser.add_argument('--cogrnn_F', action='store_true', default=True, help='whether to use F for cogrnn output')
    # attention layer after
    parser.add_argument('--attention', action='store_true', default=True,
                        help='whether to use attention on the core output')
    parser.add_argument('--attention_type', type=str, default='scaled_dot_prod',
                        choices=['additive', 'scaled_dot_prod', 'scaled_dot_prod_pos_enc'],
                        help='attention type to use')
    # d_model should be same as encoder output size
    parser.add_argument('--attention_d_model', type=int, nargs='+', default=[64, 64],
                        help="attention layer model dimensions")
    parser.add_argument('--attention_d_ff', type=int, default=128,
                        help="intermediate number of nodes for position-wise ff net")
    parser.add_argument('--attention_dropout_prob', type=float, default=0.2,
                        help='dropout probability for attention layer')
    parser.add_argument('--attention_n_heads', type=int, default=4, help='number of attention heads for each layer')

    # fc layer for critic and actor
    parser.add_argument('--critic_arch', type=int, nargs='+', default=[128], help='mlp architecture for critic')
    parser.add_argument('--critic_activations', type=str, nargs='+', default=['relu'],
                        help='activations for mlp network of critic')
    parser.add_argument('--actor_arch', type=int, nargs='+', default=[128], help='mlp architecture for actor')
    parser.add_argument('--actor_activations', type=str, nargs='+', default=['relu'],
                        help='activations for mlp network of actor')
    # agent args
    parser.add_argument('--agent', type=str, default='a2c', help='type of RL agent to use')
    parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
    parser.add_argument('--lamda', type=float, default=0.95, help='GAE discount factor')
    parser.add_argument('--horizon', type=int, default=256, help='step before an update')
    parser.add_argument('--do_gae', action='store_true', default=True,
                        help='whether to use GAE or not for advantage calculation')
    parser.add_argument('--v_coeff', type=float, default=0.5, help='weight on value loss')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='weight on entropy loss')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='maximum on the clipping of gradient norm')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--train_steps', type=int, default=10000000, help='steps to run for training')
    parser.add_argument('--test_interval', type=int, default=2000, help='training steps before performing a test')
    parser.add_argument('--test_episodes', type=int, default=100, help='episodes to run for testing')

    parser.add_argument('--log_render', action='store_true', default=False, help='whether to log rendering or not')
    parser.add_argument('--log_ratemaps', action='store_true', default=False, help='whether to log ratemaps or not')
    parser.add_argument('--num_envs', type=int, default=20, help='number of environments to use')
    # miscellaneous args

    args = parser.parse_args()

    return args


def main(args):
    with open("configs_fixed.json", 'r') as json_file:
        parsed_json = json_file.read()
    config_json = json.loads(parsed_json)  # for stationary parameters throughout the experiment

    wandb.init(mode=config_json['wandb_mode'], project=config_json['wandb_project'], group=config_json['wandb_group'],
               entity=config_json['wandb_entity'], config=args)
    wandb.config.update(config_json)

    # define metric
    wandb.define_metric('steps/*')
    wandb.define_metric('trial_number')
    wandb.define_metric('loss/*', step_metric='steps/backprop')
    wandb.define_metric('performance', step_metric='steps/global')
    wandb.define_metric('average_successful_fraction', step_metric='steps/global')
    wandb.define_metric('performance', step_metric='trial_number')
    wandb.define_metric('average_successful_fraction', step_metric='trial_number')

    # configs
    env_config = {
        'time_intervals': config_json['env_timing_stimulus'],
        'height_obs': config_json['env_height'],
        'width_obs': config_json['env_width'],
        'dt': config_json['env_dt'],
        'reward_for_steps': config_json['env_reward_steps'],
        'reward_abort': config_json['env_reward_abort'],
        'reward_incorrect': config_json['env_reward_fail'],
        'reward_correct': config_json['env_reward_correct'],
        'render_mode': config_json['env_render_mode'],

    }
    agent = None

    if torch.cuda.is_available() and config_json['device'] == 'gpu':
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and config_json['device'] == 'gpu':
        device = torch.device(
            'mps')  # for mac, it has several problems: float32, sigerror for lstm, logaddexp not supported
    else:
        device = torch.device('cpu')

    encoder_config = None
    core_config = None

    if args.encoder == 'mlp':
        encoder_config = {
            'type': 'mlp',
            'net_arch': args.mlp_arch
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
            'type': 'rnn',
            'num_layers': args.rnn_num_layers,
            'hidden_size': args.rnn_n_taus * args.conv_mlp_arch[-1],  # change it if not using convolutional encoder
            'batch_first': args.rnn_batch_first,
            'attention': {
                'present': args.attention,
                'type': args.attention_type,
                'num_layers': len(args.attention_d_model),
                'd_model': args.attention_d_model,
                'd_ff': args.attention_d_ff,
                'n_heads': args.attention_n_heads,
                'dropout': args.attention_dropout_prob
            }
        }
    elif args.core == 'lstm':
        core_config = {
            'type': 'lstm',
            'num_layers': args.lstm_num_layers,
            'hidden_size': args.lstm_n_taus * args.conv_mlp_arch[-1],  # change it if not using convolutional encoder,
            'batch_first': args.lstm_batch_first,
            'attention': {
                'present': args.attention,
                'type': args.attention_type,
                'num_layers': len(args.attention_d_model),
                'd_model': args.attention_d_model,
                'd_ff': args.attention_d_ff,
                'n_heads': args.attention_n_heads,
                'dropout': args.attention_dropout_prob
            }
        }
    elif args.core == 'cogrnn':
        core_config = {
            'type': 'cogrnn',
            'tstr_min': args.cogrnn_tstr_min,
            'tstr_max': args.cogrnn_tstr_max,
            'n_taus': args.cogrnn_n_taus,
            'k': args.cogrnn_k,
            'dt': args.cogrnn_dt,
            'g': args.cogrnn_g,
            'batch_first': args.cogrnn_batch_first,
            'alpha_mod': args.cogrnn_in_alpha,
            'F': args.cogrnn_F,
            'attention': {
                'present': args.attention,
                'type': args.attention_type,
                'num_layers': len(args.attention_d_model),
                'd_model': args.attention_d_model,
                'd_ff': args.attention_d_ff,
                'n_heads': args.attention_n_heads,
                'dropout': args.attention_dropout_prob
            }
        }

    critic_config = {
        'net_arch': [] if args.attention else args.critic_arch,
        'activations': args.critic_activations
    }
    actor_config = {
        'net_arch': [] if args.attention else args.actor_arch,
        'activations': args.actor_activations
    }

    # define vectorized env
    envs = gym.vector.SyncVectorEnv([lambda: make_env3D(env_config)] * args.num_envs)
    test_env = TempBisectionEnv3D(**env_config)
    test_env = ScaleObs(test_env)

    network = AgentNetwork(envs.single_observation_space.shape, num_actions=envs.single_action_space.n,
                           batch_size=args.num_envs, encoder_config=encoder_config, core_config=core_config,
                           critic_config=critic_config, actor_config=actor_config, device=device).to(device)

    print(network)

    count_parameters(network)
    # freeze the core
    if args.core_freeze:
        for name, parameter in network.named_parameters():
            if 'core' in name:
                parameter.requires_grad = False

    if args.agent == "a2c":
        agent_config = {
            'gamma': args.gamma,
            'lamda': args.lamda,
            'horizon': args.horizon,
            'gae': args.do_gae,
            'learning_rate': args.learning_rate,
            'v_coeff': args.v_coeff,
            'entropy_coeff': args.entropy_coeff,
            'max_grad_norm': args.max_grad_norm
        }
        agent = A2CAgent(envs, test_env, args.num_envs, model=network, agent_config=agent_config,
                         core_config=core_config, device=device, run_name=wandb.run.name, log_render=args.log_render,
                         log_ratemaps=args.log_ratemaps, log_test_info_intervals=config_json['log_test_info_intervals'],
                         log_test_info_episodes=config_json['log_test_info_episodes'])

    _ = agent.run(training_steps=args.train_steps, test_interval=args.test_interval, test_episodes=args.test_episodes)

    wandb.run.finish()


if __name__ == "__main__":
    args = parse_argument()  # not using arg_parser for the time being
    main(args)
