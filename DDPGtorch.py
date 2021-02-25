import numpy as np
import torch as T
import torch.functional as F
from replay_memory import Per_Memory, Uniform_Memory
import os
from DDPG_reg import Actor, Critic
class DDPG():
    def __init__(self,n_states, n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, PER, GAMMA,
                 tau, batch_size, save_dir):

        self.GAMMA = GAMMA
        self.batch_size = batch_size
        self.tau = tau
        self.PER = PER

        self.save_dir = save_dir

        self.actor = Actor(n_states, n_action, action_bound, layer_1_nodes, layer_2_nodes)
        self.critic = Critic(n_states, n_action,layer_1_nodes, layer_2_nodes)

        self.actor_target = Actor(n_states, n_action, action_bound, layer_1_nodes, layer_2_nodes)
        self.critic_target = Critic(n_states, n_action,layer_1_nodes, layer_2_nodes)

        if self.PER:
            self.memory = Per_Memory(capacity=100000)
        else:
            self.memory = Uniform_Memory(buffer_size=100000)

        self.sum_q = 0
        self.actor_loss = 0
        self.critic_loss = 0

    def train(self):
        # sample from memory
        if self.batch_size < self.memory.get_count:
            if self.PER:
                mem, idxs, self.isweight = self.memory.sample(self.batch_size)
            else:
                mem = self.memory.sample(self.batch_size)
            s_rep = T.tensor(np.array([_[0] for _ in mem]), dtype=T.float)
            a_rep = T.tensor(np.array([_[1] for _ in mem]), dtype=T.float)
            r_rep = T.tensor(np.array([_[2] for _ in mem]), dtype=T.float)
            s1_rep = T.tensor(np.array([_[3] for _ in mem]), dtype=T.float)
            d_rep = T.tensor(np.array([_[4] for _ in mem]), dtype=T.float)

            # Calculate critic and train
            targ_actions = self.actor_target.forward(s1_rep)
            target_q = self.critic_target.forward(s1_rep, targ_actions)
            q = self.critic(s_rep, a_rep)

            target_q = target_q.view(-1)
            
            y_i = r_rep + self.GAMMA * target_q * (1-d_rep)
            y_i = y_i.view(self.batch_size, 1)

            self.critic.optimizer.zero_grad()
            critic_loss = T.nn.functional.mse_loss(y_i, q)
            critic_loss.backward()
            self.critic.optimizer.step()

            # Calculate actor and train
            self.actor.optimizer.zero_grad()
            actions = self.actor(s_rep)
            actor_loss = T.mean(-self.critic(s_rep, actions))
            actor_loss.backward()
            self.actor.optimizer.step()

            # update target network
            self.update_target_network(self.tau)

    def update_target_network(self, tau=.001):
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.actor_target.named_parameters()
        target_critic_params = self.critic_target.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.critic_target.load_state_dict(critic_state_dict)
        self.actor_target.load_state_dict(actor_state_dict)
        # actor = self.actor.named_parameters()
        # actor_targ_params = self.actor_target.named_parameters()
        
        # actor_dict = dict(actor)
        # actor_targ_dict = dict(actor_targ_params)

        # for name in actor_dict:
        #     actor_dict[name] = tau*actor_dict[name].clone() +\
        #                         (1-tau)*actor_targ_dict[name].clone()

        # critic = self.critic.named_parameters()
        # critic_targ_params = self.critic_target.named_parameters()
        
        # critic_dict = dict(critic)
        # critic_targ_dict = dict(critic_targ_params)

        # for name in critic_dict:
        #     critic_dict[name] = tau*critic_dict[name].clone() +\
        #                         (1-tau)*critic_targ_dict[name].clone()

        # self.actor_target.load_state_dict(actor_dict)
        # self.critic_target.load_state_dict(critic_dict)

    def action(self, state):
        self.actor.eval()
        state = T.tensor([state], dtype=T.float)
        act = self.actor.forward(state)

        self.actor.train()
        return act.cpu().detach().numpy()[0]

    # def save_model(self):
    #     self.actor.save_weights(os.path.join(self.save_dir, self.actor.model_name))
    #     self.critic.save_weights(os.path.join(self.save_dir, self.critic.model_name))
    #     self.actor_target.save_weights(os.path.join(self.save_dir, self.actor_target.model_name))
    #     self.critic_target.save_weights(os.path.join(self.save_dir, self.critic_target.model_name))

    # def load_model(self):
    #     self.actor.load_weights(os.path.join(self.save_dir, self.actor.model_name))
    #     self.critic.load_weights(os.path.join(self.save_dir, self.critic.model_name))
    #     self.actor_target.load_weights(os.path.join(self.save_dir, self.actor_target.model_name))
    #     self.critic_target.load_weights(os.path.join(self.save_dir, self.critic_target.model_name))
