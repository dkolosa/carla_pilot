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

        self.update_target_network()

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
                mem, idxs, isweight = self.memory.sample(self.batch_size)
                isweight = T.tensor(isweight, dtype=T.float).to(self.actor.device)
            else:
                mem = self.memory.sample(self.batch_size)
            s_rep = T.tensor(np.array([_[0] for _ in mem]), dtype=T.float).to(self.actor.device)
            a_rep = T.tensor(np.array([_[1] for _ in mem]), dtype=T.float).to(self.actor.device)
            r_rep = T.tensor(np.array([_[2] for _ in mem]), dtype=T.float).to(self.actor.device)
            s1_rep = T.tensor(np.array([_[3] for _ in mem]), dtype=T.float).to(self.actor.device)
            d_rep = T.tensor(np.array([_[4] for _ in mem]), dtype=T.float).to(self.actor.device)

            self.critic.eval()
            self.actor.eval()
            self.actor_target.eval()
            self.critic_target.eval()

            # Calculate critic and train
            targ_actions = self.actor_target.forward(s1_rep)
            target_q = self.critic_target.forward(s1_rep, targ_actions)
            q = self.critic.forward(s_rep, a_rep)

            target_q = target_q.view(-1)
            y_i = r_rep + self.GAMMA * target_q * (1-d_rep)
            y_i = y_i.view(self.batch_size, 1)

            if self.PER:
                td_error = y_i - q
                update_error = T.abs(td_error).cpu().detach().numpy()
                for i in range(self.batch_size):
                    self.memory.update(idxs[i], update_error[i])

            self.critic.train()
            self.critic.optimizer.zero_grad()
            if not self.PER:
                critic_loss = T.nn.functional.mse_loss(y_i, q)
            else:
                critic_loss = T.mean(T.square(td_error) * isweight)
            critic_loss.backward()
            self.critic.optimizer.step()
            self.critic.eval()

            # Calculate actor and train
            self.actor.optimizer.zero_grad()
            actions = self.actor.forward(s_rep)
            self.actor.train()
            actor_loss = T.mean(-self.critic.forward(s_rep, actions))
            actor_loss.backward()
            self.actor.optimizer.step()
            self.actor.eval()

            # update target network
            self.update_target_network(self.tau)

    def update_target_network(self, tau=.001):
        actor = self.actor.named_parameters()
        actor_targ_params = self.actor_target.named_parameters()
        
        actor_dict = dict(actor)
        actor_targ_dict = dict(actor_targ_params)

        for name in actor_dict:
            actor_dict[name] = tau*actor_dict[name].clone() +\
                                (1-tau)*actor_targ_dict[name].clone()

        critic = self.critic.named_parameters()
        critic_targ_params = self.critic_target.named_parameters()
        
        critic_dict = dict(critic)
        critic_targ_dict = dict(critic_targ_params)

        for name in critic_dict:
            critic_dict[name] = tau*critic_dict[name].clone() +\
                                (1-tau)*critic_targ_dict[name].clone()

        self.actor_target.load_state_dict(actor_dict)
        self.critic_target.load_state_dict(critic_dict)

    def action(self, state):
        self.actor.eval()
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        act = self.actor.forward(state).to(self.actor.device)

        self.actor.train()
        return act.cpu().detach().numpy()[0]

    def preprocess_image(self, image):

    def load_model(self):
        self.actor.load_state_dict(T.load(os.path.join(self.save_dir, self.actor.model_name)))
        self.critic.load_state_dict(T.load(os.path.join(self.save_dir, self.critic.model_name)))
        self.actor_target.load_state_dict(T.load(os.path.join(self.save_dir, self.actor_target.model_name)))
        self.critic_target.load_state_dict(T.load(os.path.join(self.save_dir, self.critic_target.model_name)))

    def save_model(self):
        self.actor.save_model()
        self.critic.save_model()
