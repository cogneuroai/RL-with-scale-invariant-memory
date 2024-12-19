import torch
import numpy as np
import pandas as pd
import os
import wandb

torch.set_default_dtype(torch.float64)


# A2C class
class A2CAgent:
    def __init__(self, env, test_env, num_envs, model, agent_config, core_config, run_name, device, log_render, log_ratemaps, log_test_info_intervals, log_test_info_episodes):
        '''
        constructor for A2C agent
        :param env: training environment instance
        :param test_env: test environment instance
        :param num_envs: number of environments used in the training environment
        :param model: model instance for the agent
        :param agent_config: dictionary containing agent configuration
        :param core_config: dictionary containing core configuration
        '''
        self.env = env
        self.test_env = test_env
        self.num_envs = num_envs
        self.model = model
        self.learning_rate = agent_config['learning_rate']
        self.gamma = agent_config['gamma']
        self.lamda = agent_config['lamda']
        self.horizon = agent_config['horizon']
        self.v_coeff = agent_config['v_coeff']
        self.entropy_coeff = agent_config['entropy_coeff']
        self.max_grad_norm = agent_config['max_grad_norm']
        self.do_gae = agent_config['gae']
        self.core_config = core_config
        self.device = device
        self.optimizer = None
        self.critic_criterion = None

        self.run_name = run_name
        self.f_name = str(run_name)
        self.log_render = log_render
        self.log_ratemaps = log_ratemaps
        self.test_block = 0
        self.test_successful_trial_num = 0
        self.test_trial_num = 0
        self.log_trials = pd.DataFrame(columns=["block_id", "trial_id", "stimulus", "success"]) # to tabulate the data
        self.log_test_info_intervals = log_test_info_intervals
        self.log_test_info_episodes = log_test_info_episodes
        self.trial_number = 0


    def calc_gae(self, rewards, state_vals, dones, R):
        '''

        :param rewards:
        :param state_vals:
        :param dones:
        :param R:
        :return: GAE and target
        '''
        # GAE calculation: stale
        targets = torch.zeros_like(rewards).to(self.device)
        advs = torch.zeros_like(rewards).to(self.device)
        gae_cumulative = 0
        # converting rewards and dones dim to (horizon, batch, 1)
        for r, v, d, i in list(zip(rewards, state_vals, dones, range(self.horizon)))[::-1]:
            delta = r + (self.gamma * R * ~(d)) - v
            gae_cumulative = (self.lamda * self.gamma * ~(d) * gae_cumulative) + delta
            R = v
            advs[i] = gae_cumulative
            targets[i] = gae_cumulative + v

        return advs, targets

    def calc_adv(self, rewards, state_vals, dones, R):
        '''

        :param rewards:
        :param state_vals:
        :param dones:
        :param R:
        :return: advantage and target
        '''
        # discounted rewards
        targets = torch.zeros_like(rewards).to(self.device)
        # converting rewards and dones dim to (horizon, batch, 1)
        for r, d, i in list(zip(rewards, dones, range(self.horizon)))[::-1]:
            R = r + (self.gamma * R * ~(d))
            targets[i] = R
        # advantage calculation
        advs = targets - state_vals  # may need to normalize advantages

        return advs, targets

    def train(self, state_vals, act_probs_a, rewards, dones, entropies, next_state, h_next, done_next):
        '''

        :param state_vals:
        :param act_probs_a:
        :param rewards:
        :param dones:
        :param entropies:
        :param next_state:
        :return: critic and actor loss
        '''
        # calculate advantage, target
        R = self.model(torch.tensor(next_state[:, None, :]).to(self.device), h_next, torch.tensor(done_next[:, None, None]).to(self.device))[0]
        if self.do_gae:
            advantages, disc_rewards = self.calc_gae(rewards, state_vals, dones, R)
        else:
            advantages, disc_rewards = self.calc_adv(rewards, state_vals, dones, R)

        # loss calculation
        # converting tensors from (time, batch, 1) to (batch, time, 1)
        critic_loss = self.critic_criterion(torch.permute(state_vals, (1, 0, 2)), torch.permute(disc_rewards, (1, 0, 2)))
        neg_log_pa = -torch.permute(act_probs_a, (1, 0, 2)).log()
        actor_loss = torch.mean(neg_log_pa * torch.permute(advantages.detach(), (1, 0, 2)))
        total_loss = (self.v_coeff * critic_loss) + actor_loss - (self.entropy_coeff * torch.mean(torch.permute(entropies, (1, 0, 2))))

        self.optimizer.zero_grad()
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()  # performing backpropagation and updating parameters with the gradients
        return critic_loss, actor_loss


    def test(self, test_episodes):
        '''

        :param test_episodes:
        :return: average evaluation reward
        '''

        self.model.eval()
        self.test_block += 1
        self.test_trial_num = 0
        self.test_successful_trial_num = 0
        with torch.no_grad():
            test_num_envs = 1
            eval_rewards = []
            for test_ep in range(test_episodes):
                self.test_trial_num += 1
                test_state,_ = self.test_env.reset()
                #test_stim = self.test_env.time_interval
                test_done = np.zeros((test_num_envs), dtype=np.bool_)

                if isinstance(self.model.init_hidden(test_num_envs), tuple):
                    test_h = (self.model.init_hidden(test_num_envs)[0].clone(), self.model.init_hidden(test_num_envs)[1].clone())  # should be already in device
                else:
                    test_h = self.model.init_hidden(test_num_envs).clone()

                eval_ep_reward = 0

                while not test_done:
                    _, test_act_prob, encoder_output, core_output, test_h = self.model(torch.tensor(test_state[None, None, :]).to(self.device), h=test_h, done=torch.tensor(test_done)[:, None, None].to(self.device))
                    #test_act = torch.argmax(test_act_prob)     # greedy
                    dist = torch.distributions.Categorical(probs=test_act_prob)
                    test_act = dist.sample()  # stochastic
                    test_next_state, test_reward, terminated, truncated, info = self.test_env.step(test_act.cpu().numpy()[0])
                    test_done = terminated or truncated
                    # converting done to a numpy array
                    test_done = np.array(test_done).reshape(-1, )
                    eval_ep_reward += test_reward
                    test_state = test_next_state

                # checking if the previous trial was a success or not
                self.test_successful_trial_num += int(info['decision'])
                #successful_trial = info['decision']
                # self.log_trials = pd.concat([self.log_trials, pd.DataFrame([[self.test_block, self.test_trial_num,  test_stim, successful_trial]], columns=["block_id", "trial_id", "stimulus", "success"])])
                eval_rewards.append(eval_ep_reward)

        # self.log_trials.to_csv("results/" + self.f_name + '.csv', index=False)
        self.model.train()

        return np.mean(eval_rewards)

    def run(self, training_steps, test_interval, test_episodes):
        '''
        collect trajectories first, update the parameters, then after certain interval tests
        :param learning_rate:
        :param training_steps:
        :param test_interval:
        :param test_episodes:
        :return: a list of average test rewards throughout the whole training steps
        '''

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.critic_criterion = torch.nn.MSELoss(reduction="mean")

        os.makedirs("results/", exist_ok=True)
        os.makedirs("models/", exist_ok=True)

        test_rewards = [] #to collect average eval rewards
        # obtain trajectories
        state, _ = self.env.reset()
        done = np.zeros((self.num_envs), dtype=np.bool_)
        if isinstance(self.model.init_h, tuple):
            h = (self.model.init_h[0].clone(), self.model.init_h[1].clone())
        else:
            h = self.model.init_h.clone()

        ep_reward = np.zeros((self.num_envs, 1))
        step = 0
        t_step = 0
        global_step = 0
        state_vals = torch.zeros((self.horizon, self.num_envs, 1)).to(self.device)
        act_probs_a = torch.zeros((self.horizon, self.num_envs, 1)).to(self.device)
        rewards = torch.zeros((self.horizon, self.num_envs, 1)).to(self.device)
        dones = torch.zeros((self.horizon, self.num_envs, 1), dtype=torch.bool).to(self.device)
        entropies = torch.zeros((self.horizon, self.num_envs, 1)).to(self.device)

        while global_step <= training_steps:
            # maybe modify the obs shape
            state = torch.tensor(state[:, None, :]) # get an observation of shape (batch, time, feat)
            v, act_prob, encoder_output, core_ouput, h = self.model(state.to(self.device), h, done=torch.tensor(done[:, None, None]).to(self.device))  # state dimension (batch, feat_size), act_prob have dim (batch, num_action)
            dist = torch.distributions.Categorical(probs=act_prob)
            entropy = dist.entropy()  # dimension (batch)
            entropies[t_step] = entropy[:, None]
            act = dist.sample()  # act should have dimension (batch)
            act_prob_a = act_prob[range(act_prob.shape[0]), act]  # dimension (batch)
            state_vals[t_step] = v
            act_probs_a[t_step] = act_prob_a[:, None]
            next_state, reward, terminated, truncated, info = self.env.step(act.detach().cpu().numpy())  # everything should have dimension (batch, ...)
            done = terminated | truncated
            self.trial_number += torch.sum(torch.tensor(done)).item()
            ep_reward += reward[:, None]
            rewards[t_step] = torch.tensor(reward)[:, None]
            dones[t_step] = torch.tensor(done)[:, None]

            t_step += 1

            state = next_state

            if(((step+1) % self.horizon)==0):

                # perform training
                critic_loss, actor_loss = self.train(state_vals, act_probs_a, rewards, dones, entropies, next_state, h, done)
                log_metric_dict = {
                    "steps/backprop": step/self.horizon,
                    "loss/critic": critic_loss,
                    "loss/actor": actor_loss,
                }
                wandb.log(log_metric_dict, commit=False)

                # detaching to not retain graph
                if isinstance(h, tuple):
                    hx, cx = h
                    hx = hx.detach()  # detach to not retain graph
                    cx = cx.detach()
                    h = (hx, cx)
                else:
                    h = h.detach()

                # clearing the lists
                state_vals = torch.zeros((self.horizon, self.num_envs, 1)).to(self.device)
                act_probs_a = torch.zeros((self.horizon, self.num_envs, 1)).to(self.device)
                rewards = torch.zeros((self.horizon, self.num_envs, 1)).to(self.device)
                dones = torch.zeros((self.horizon, self.num_envs, 1), dtype=torch.bool).to(self.device)
                entropies = torch.zeros((self.horizon, self.num_envs, 1)).to(self.device)
                t_step = 0

            # evaluate
            if (global_step % test_interval) == 0:  # do the evaluation on the first step
                # save the model
                if(global_step % (5*test_interval)) == 0:
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }, "models/" + self.f_name + '_model.pt')

                    # save the model as an artifact
                    wandb.run.log_artifact("models/" + self.f_name + '_model.pt', type='model')

                avg_evaluation_rewards = self.test(test_episodes)
                test_success_fraction = self.test_successful_trial_num / self.test_trial_num
                log_metric_dict = {
                    "steps/global": global_step,
                    "trial_number": self.trial_number,
                    "performance": avg_evaluation_rewards,
                    "average_successful_fraction": test_success_fraction
                }
                wandb.log(log_metric_dict, commit=True)

                test_rewards.append(avg_evaluation_rewards)
                print("Step: {} average reward {} and successful trial fraction {}".format(global_step, avg_evaluation_rewards, test_success_fraction))

            global_step += self.num_envs  # for keeping track of the global steps (because of number of environments)
            step += 1  # keeping track of the steps

        return test_rewards