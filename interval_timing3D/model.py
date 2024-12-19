# loading the encoder and core architecture to build a model
import time
import torch
from encoders import mlp_encoder, conv_mlp_encoder, build_critic, build_actor
from cores import lstm_core, rnn_core, cogrnn_core
from attention import AttentionLayer

torch.set_default_dtype(torch.float64)


class AgentNetwork(torch.nn.Module):
    def __init__(self, obs_size, num_actions, batch_size, encoder_config, core_config, critic_config, actor_config, device):
        super(AgentNetwork, self).__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.encoder_config = encoder_config
        self.core_config = core_config
        self.attention_config = core_config['attention']
        self.encoder_output_node = None
        self.core_output_node = None
        self.device = device
        self.batch_size = batch_size

        if(encoder_config['type'] == 'mlp'):
            self.encoder = mlp_encoder(input_size=obs_size, net_arch=encoder_config["net_arch"].copy())
            self.encoder_output_node = encoder_config["net_arch"][-1]

        elif(encoder_config['type'] == 'conv'):
            self.encoder = conv_mlp_encoder(input_shape=[obs_size[0], obs_size[1]], in_channel=1, conv_net_config=encoder_config["conv_config"], mlp_net_config=encoder_config["mlp_config"])
            self.encoder_output_node = encoder_config["mlp_config"]["net_arch"][-1]

        if (core_config["type"] == 'rnn'):
            self.core = rnn_core(input_size=self.encoder_output_node, net_config=core_config)
            self.core_output_node = core_config["hidden_size"]
        elif (core_config["type"] == 'lstm'):
            self.core = lstm_core(input_size=self.encoder_output_node, net_config=core_config)
            self.core_output_node = core_config["hidden_size"]
        elif (core_config["type"] == "cogrnn"):
            self.core = cogrnn_core(core_config).to(self.device)
            self.core_output_node = self.encoder_output_node * core_config['n_taus']

        if self.attention_config["present"]:
            # for the first attention layer the input dim should be encoder size
            attention_input_dims = [self.encoder_output_node] + self.attention_config['d_model'][:-1]
            self.attentions = torch.nn.ModuleList([
                AttentionLayer(
                    input_dim=[attention_input_dims[i]]*3,
                    output_dim=self.attention_config['d_model'][i],
                    n_heads=self.attention_config['n_heads'],
                    d_ff=self.attention_config['d_ff'],
                    dropout=self.attention_config['dropout'],
                ) for i in range(self.attention_config['num_layers'])
            ])
            self.core_output_node = self.attention_config['d_model'][-1]

        self.post_rnn = torch.nn.ReLU()     #torch.nn.Identity()
        # have option to define and actor and critic layer
        self.critic = build_critic(self.core_output_node, critic_config)
        self.actor = build_actor(self.core_output_node, actor_config, num_actions)
        self.actor_activation = torch.nn.Softmax(dim=-1)
        self.init_h = self.init_hidden(batch_size=batch_size)

    def init_hidden(self, batch_size):
        '''
        initialize the hidden states
        :param batch_size:
        :param horizon:
        :param feat_num:
        :return:
        '''
        init_h = None
        if self.core_config["type"] == "cogrnn":
            init_h = torch.zeros(batch_size, self.encoder_output_node, self.core_config['n_taus'] + (2 * self.core_config['k'])).to(self.device)
            init_h = init_h.log()
        elif self.core_config["type"] == "rnn":
            init_h = torch.zeros(self.core_config["num_layers"], batch_size, self.core_config["hidden_size"]).to(self.device)
        elif self.core_config["type"] == "lstm":
            init_h = (torch.zeros(self.core_config["num_layers"], batch_size, self.core_config["hidden_size"]).to(self.device), torch.zeros(self.core_config["num_layers"], batch_size, self.core_config["hidden_size"]).to(self.device))
        return init_h

    def forward(self, x, h, done):
        # x should have dimension (batch, time, height, width, channel)
        # remove the time dimension, permute and get the channel dimension at the beginning, add the time dimension
        obs = x[:, 0, :, :, :].clone()
        obs = torch.permute(obs, (0, 3, 1, 2))
        #print(obs.shape)
        encoder_outputs = self.encoder(obs.double())
        encoder_outputs = encoder_outputs[:, None, :] #adding the time dimension
        #print(encoder_outputs.shape)
        core_outputs = None
        mem_out = None
        if self.core_config['type'] == "rnn":
            core_outputs = []
            # permuting the time dimension as the first one to unwrap in time dimension
            for encoder_output, d in zip(torch.permute(encoder_outputs, (1, 0, 2)), torch.permute(done, (1, 0, 2))):
                core_output, h = self.core(encoder_output[:, None, :], (~d).view(1, -1, 1) * h)
                core_outputs.append(core_output[:, 0, :])
            core_outputs = torch.stack(core_outputs, dim=1)     # dim b, t, h_size
            mem_out = torch.flatten(core_outputs, 0, 1).clone()     # merging batch and time dimension
            if self.attention_config['present']:
                if self.attention_config['type'] == "scaled_dot_prod":
                    b, t, h_size = core_outputs.shape
                    core_outputs = core_outputs.view(b, t, self.encoder_output_node, -1)
                    attention_out = encoder_outputs.clone()  # to have shape b, t, f, n_tau
                    for attn_layer in self.attentions:
                        attention_out = attn_layer(attention_out, core_outputs)
                    core_outputs = attention_out  # dim: b,1,d_model[-1]
                    # print(core_outputs.shape)

        elif self.core_config['type'] == "lstm":
            core_outputs = []
            # permuting the time dimension as the first one to unwrap in time dimension
            for encoder_output, d in zip(torch.permute(encoder_outputs, (1, 0, 2)), torch.permute(done, (1, 0, 2))):
                core_output, h = self.core(encoder_output[:, None, :], ((~d).view(1, -1, 1) * h[0], (~d).view(1, -1, 1) * h[1]))
                core_outputs.append(core_output[:, 0, :])
            core_outputs = torch.stack(core_outputs, dim=1)
            mem_out = torch.flatten(core_outputs, 0, 1).clone()     # merging batch and time dimension and cloning
            if self.attention_config['present']:
                if self.attention_config['type'] == "scaled_dot_prod":
                    b, t, h_size = core_outputs.shape
                    core_outputs = core_outputs.view(b, t, self.encoder_output_node, -1)[:, 0, :, :].transpose(1, 2)
                    attention_out = encoder_outputs.clone()    # to have shape b, t, f, n_tau
                    for attn_layer in self.attentions:
                        attention_out = attn_layer(attention_out, core_outputs)
                    core_outputs = attention_out    # dim: b,1,d_model[-1]
                    #print(core_outputs.shape)
        elif self.core_config['type'] == "cogrnn":
            # encoder should have dimension batch, time, feat
            done_indices = torch.flatten((torch.flatten(done)==True).nonzero()) # obtaining the done indices
            h[done_indices] = self.init_h[done_indices].clone() # replacing the terminated env's hidden state
            if self.core_config['alpha_mod']:
                f_tilde, h, F = self.core(torch.zeros_like(encoder_outputs), h, encoder_outputs)  # core output has dimension batch, time, feat, n_taus
            else:
                f_tilde, h, F = self.core(encoder_outputs, h, torch.ones_like(encoder_outputs))

            if self.core_config['F']:
                core_outputs = F
            else:
                core_outputs = f_tilde

            mem_out = core_outputs.clone()
            if self.attention_config['present']:
                if self.attention_config['type'] == "scaled_dot_prod":
                    attention_out = encoder_outputs.clone()
                    for attn_layer in self.attentions:
                        attention_out = attn_layer(attention_out, core_outputs)
                    core_outputs = attention_out    # dim: b,1,d_model[-1]
                    #print(core_outputs.shape)
            else:
                core_outputs = torch.flatten(core_outputs, -2)  # merging (feat and n_taus) and (batch and time) dimension

        post_rnn = self.post_rnn(core_outputs)
        critic_output = self.critic(post_rnn)
        actor_output = self.actor_activation(self.actor(post_rnn))

        return critic_output.squeeze(1), actor_output.squeeze(1), encoder_outputs, mem_out, h