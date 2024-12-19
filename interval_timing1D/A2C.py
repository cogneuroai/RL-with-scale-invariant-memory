import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from interval_timing import IntervalTiming1D
import json
from CogRNN import CogRNN
import argparse
import os
import random

def parse_argument():

    parser = argparse.ArgumentParser()
    parser.add_argument('--core', type=str, default='rnn', choices=['rnn', 'lstm', 'cogrnn'], help='core type to use')
    parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
    parser.add_argument('--model_fname', type=str, default='test_interval_timing', help='file name of model to be saved')
    parser.add_argument('--env_scale', type=int, default=1, help='environment scale')

    arguments = parser.parse_args()

    return arguments

class ActorCriticRNN(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(ActorCriticRNN, self).__init__()
        self.core = nn.RNN(input_dim, hidden_dim, batch_first=True)
        #add postprocessor
        self.postprocessor = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU())
        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x, h):
        #x = self.preprocessor(x)    # preprocessor output
        x, h = self.core(x.unsqueeze(1), h)   # unsqueeze add the time dimension
        x = x.squeeze(1)    # getting rid of the time dimension
        x = self.postprocessor(x)   # postprocessor output
        policy_dist = self.actor(x)
        value = self.critic(x)
        return policy_dist, value, h

class ActorCriticLSTM(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(ActorCriticLSTM, self).__init__()
        self.core = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        #add postprocessor
        self.postprocessor = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU())
        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x, h):
        x, h = self.core(x.unsqueeze(1), h)   # unsqueeze add the time dimension
        x = x.squeeze(1)    # getting rid of the time dimension
        x = self.postprocessor(x)   # postprocessor output
        policy_dist = self.actor(x)
        value = self.critic(x)
        return policy_dist, value, h

class ActorCriticCogRNN(nn.Module):
    def __init__(self, action_dim):
        super(ActorCriticCogRNN, self).__init__()
        self.core = CogRNN(tstr_min=1.0, tstr_max=1000, n_taus=80, k=8, dt=1.0, g=1, batch_first=True)
        #add postprocessor
        self.postprocessor = nn.Sequential(nn.Linear(80, 64), nn.ReLU())
        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x, h):
        x, h, _ = self.core(x.unsqueeze(1), h)   # unsqueeze add the time dimension
        x = x.squeeze(1)    # getting rid of the time dimension
        x = torch.flatten(x, -2)
        #print(x.shape)
        x = self.postprocessor(x)   # postprocessor output
        policy_dist = self.actor(x)
        value = self.critic(x)
        return policy_dist, value, h

def compute_returns(rewards, dones, next_value, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * (1 - dones[step])
        returns.insert(0, R)
    return returns


def train(env, model, optimizer, num_episodes=1000, gamma=0.99, ckpt_path=None):
    performance_log = {
        'episodes': [],
        'average_scores': [],
    }
    ep_rewards = []
    for episode in range(num_episodes):
        state, info = env.reset(seed=random.randint(0, 1000))     # seed?
        '''
        if isinstance(model.core, nn.LSTM):
            #hidden_dim = model.lstm.hidden_size
            h = None
        else:
            h = None
        '''
        h = None
        log_probs = []
        values = []
        rewards = []
        dones = []

        while True:
            state = torch.FloatTensor(state).unsqueeze(0)
            policy_dist, value, h = model(state, h)
            action = np.random.choice(len(policy_dist.detach().numpy().squeeze()), p=policy_dist.detach().numpy().squeeze())
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)

            state = next_state

            if done:
                break

        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        _, next_value, _ = model(next_state, h)
        returns = compute_returns(rewards, dones, next_value, gamma)

        log_probs = torch.stack(log_probs)
        returns = torch.FloatTensor(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_rewards.append(sum(rewards))
        if (episode+1)%50==0:
            print(f"Episode {episode + 1}, Loss: {loss.item()}, Total Reward: {np.mean(ep_rewards[-100:])}")
            performance_log['episodes'].append(episode + 1)
            performance_log['average_scores'].append(np.mean(ep_rewards[-100:]))
            torch.save(performance_log, ckpt_path)


if __name__ == "__main__":
    parsed_args = parse_argument()
    with open("configs_fixed.json", 'r') as json_file:
        parsed_json = json_file.read()
    config_json = json.loads(parsed_json)  # for stationary parameters throughout the experiment

    model_ckpt_folder = 'models/'
    os.makedirs(model_ckpt_folder, exist_ok=True)
    model_ckpt_filename = parsed_args.model_fname
    model_ckpt_path = model_ckpt_folder + model_ckpt_filename + '_model.pt'

    env_configs = {
            'height': config_json['env_height'],
            'width': config_json['env_width'],
            'stimuli': config_json['env_stimuli'],
            'scale': parsed_args.env_scale,
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

    print(env.observation_space.shape, env.action_space.n)

    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    torch.manual_seed(random.randint(0, 1000))  # random.randint(0, 1000)
    model = None
    if parsed_args.core == 'rnn':
        model = ActorCriticRNN(input_dim, action_dim)
    elif parsed_args.core == 'lstm':
        model = ActorCriticLSTM(input_dim, action_dim)
    elif parsed_args.core == 'cogrnn':
        model = ActorCriticCogRNN(input_dim, action_dim)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(env, model, optimizer, num_episodes=50000, gamma=parsed_args.gamma, ckpt_path=model_ckpt_path)

