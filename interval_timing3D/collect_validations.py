import wandb
import json
import torch
import numpy as np
import os
from env import IntervalTiming
from model import AgentNetwork
from agents import A2CAgent
import pickle, gzip


def validations(agent, test_episodes):
    '''

    :param agent:
    :param test_episodes:
    :return:
    '''
    agent.model.eval()

    with torch.no_grad():
        test_num_envs = 1
        test_info_whole = []
        for test_ep in range(test_episodes):
            test_state = agent.test_env.reset()
            test_gt = agent.test_env.gt[-1]
            test_stim = agent.test_env.timing['stim']
            test_done = np.zeros((test_num_envs), dtype=np.bool_)
            # test_h = self.model.init_hidden(test_num_envs)

            if isinstance(agent.model.init_hidden(test_num_envs), tuple):
                test_h = (agent.model.init_hidden(test_num_envs)[0].clone(),
                          agent.model.init_hidden(test_num_envs)[1].clone())  # should be already in device
            else:
                test_h = agent.model.init_hidden(test_num_envs).clone()

            eval_ep_reward = 0
            mem_signals = []  # to store output from cores (rnn, lstm, cogrnn)

            while not test_done:
                _, test_act_prob, encoder_output, core_output, test_h = agent.model(
                    torch.tensor(test_state[None, None, :]).to(agent.device), h=test_h,
                    done=torch.tensor(test_done)[:, None, None].to(agent.device))
                test_act = torch.argmax(test_act_prob)
                test_next_state, test_reward, _, info = agent.test_env.step(test_act.cpu().numpy())
                test_done = info['new_trial']
                # converting done to a numpy array
                test_done = np.array(test_done).reshape(-1, )
                mem_signals.append(core_output[0].cpu().numpy().flatten())
                eval_ep_reward += test_reward
                test_state = test_next_state

            # checking if the previous trial was a success or not
            if test_gt == test_act:
                agent.test_successful_trial_num += 1
                successful_trial = True
            else:
                successful_trial = False

            test_info = {
                'trial_stimulus': test_stim,
                'trial_gt': test_gt,
                'trial_reward': eval_ep_reward,
                'trial_last_action': test_act.cpu().numpy(),
                'trial_success': successful_trial,
                'mem_signals': np.array(mem_signals).T
            }
            test_info_whole.append(test_info)

    return test_info_whole


if __name__ == '__main__':
    test_episodes = 500 # number of episodes to validate
    os.makedirs("postprocessing/data/", exist_ok=True)
    # using wandb api load the run
    api = wandb.Api()
    with open("configs_performance.json", 'r') as json_file:
        parsed_json = json_file.read()
    config_json = json.loads(parsed_json)

    entity = config_json['entity']
    project = config_json['project']


    logs = []  # to store the test info along with activations across different instances of the same network and agent configs
    for core in config_json['cores'].keys():
        run_ids = config_json['cores'][core]['run_ids']
        # versions of the model to download
        versions = config_json['cores'][core]['versions']
        f_name = config_json['cores'][core] + '_validation'
        for run_id in run_ids:
            print('Run id: {}'.format(run_id))
            run = api.run(entity + '/' + project + '/' + run_id)
            # download the checkpoints using artifacts
            for v in versions:
                artifact = api.artifact(entity + '/' + project + '/' + 'run-' + run.id + '-' + run.name + '_model.pt:v'+str(v))
                art_dir = artifact.download()
                # print(art_dir)
                # load the run config: to build the configs for env, model, and agent
                configs = run.config
                core_config = None
                agent_config = None
                agent = None

                env_config = {
                    'dt': configs['env_dt'],
                    'rewards': {
                        'abort': configs['env_reward_abort'],
                        'correct': configs['env_reward_correct'],
                        'fail': configs['env_reward_fail']
                    },
                    'timing': {
                        'fixation': (3 * configs['env_dt']) + configs['env_dt'],
                        'delay': 0,
                        'decision': configs['env_dt']
                    },
                    'stimuli': configs['env_timing_stimulus'],
                    # 'stimuli': [2800, 3100, 3400, 4200, 4600, 5000],
                    'num_per_group': configs['env_num_per_group']
                }

                encoder_config = {
                    'type': 'mlp',
                    'net_arch': configs['mlp_arch']
                }

                if configs['core'] == 'rnn':
                    core_config = {
                        "type": 'rnn',
                        "num_layers": configs['rnn_num_layers'],
                        "hidden_size": configs['rnn_hidden_size'],
                        "batch_first": configs['rnn_batch_first']
                    }
                elif configs['core'] == 'lstm':
                    core_config = {
                        "type": 'lstm',
                        "num_layers": configs['lstm_num_layers'],
                        "hidden_size": configs['lstm_hidden_size'],
                        "batch_first": configs['lstm_batch_first']
                    }
                elif configs['core'] == 'cogrnn':
                    core_config = {
                        "type": 'cogrnn',
                        'tstr_min': configs['cogrnn_tstr_min'],
                        'tstr_max': configs['cogrnn_tstr_max'],
                        'n_taus': configs['cogrnn_n_taus'],
                        'k': configs['cogrnn_k'],
                        'dt': configs['cogrnn_dt'],
                        'g': configs['cogrnn_g'],
                        'batch_first': configs['cogrnn_batch_first'],
                        'F': configs['cogrnn_F'],
                    }

                agent_config = {
                    'gamma': configs['gamma'],
                    'lamda': configs['lamda'],
                    'horizon': configs['horizon'],
                    'gae': configs['do_gae'],
                    'learning_rate': configs['learning_rate'],
                    'v_coeff': configs['v_coeff'],
                    'entropy_coeff': configs['entropy_coeff'],
                }

                device = torch.device('cpu')

                # define the test env
                test_env = IntervalTiming(**env_config)
                # define the model
                network = AgentNetwork(test_env.observation_space.shape[0] * test_env.observation_space.shape[1],
                                       num_actions=test_env.action_space.n, batch_size=configs['num_envs'],
                                       encoder_config=encoder_config, core_config=core_config, device=device).to(device)

                # load the checkpoints from .pt
                checkpoints = torch.load(art_dir + '/' + run.name + '_model.pt')
                # load_state_dict the model with the checkpoints
                network.load_state_dict(checkpoints['model_state_dict'])

                # define the agent
                agent = A2CAgent(test_env, test_env, configs['num_envs'], model=network, agent_config=agent_config,
                                 core_config=core_config, device=device, run_name=run.name, log_ratemaps=configs['log_ratemaps'])

                # run the validations: define the agent and use/write the/a test function
                test_info = validations(agent, test_episodes=test_episodes)

                logs.append(test_info)
                print('done accumulating test info.')

        pickle.dump(logs, gzip.open("postprocessing/data/" + f_name + '_test_logs.pkl.gz', 'wb'))  # save the test info logs



















