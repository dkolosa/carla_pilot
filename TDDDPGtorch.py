import numpy as np
import torch as T
import torch.functional as F
from replay_memory import Uniform_Memory
import os
from model_torch_vision import Actor, Critic


class TDDDPG():
    def __init__(self,n_states, n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, GAMMA,
                 tau, batch_size, save_dir, policy_update_delay=2):

        self.GAMMA = GAMMA
        self.batch_size = batch_size
        self.tau = tau
        self.policy_update_delay = policy_update_delay

        self.save_dir = save_dir

        self.actor = Actor(n_states, n_action, action_bound,batch_size, layer_1_nodes, layer_2_nodes)
        self.critic = Critic(n_states, n_action,layer_1_nodes, layer_2_nodes)
        self.critic_delay = Critic(n_states, n_action,layer_1_nodes, layer_2_nodes,checkpt='critic-delay')
        
        self.actor_target = Actor(n_states, n_action, action_bound, layer_1_nodes, layer_2_nodes)
        self.critic_target = Critic(n_states, n_action,layer_1_nodes, layer_2_nodes)
        self.critic_target_delay = Critic(n_states, n_action,layer_1_nodes, layer_2_nodes,checkpt='actor_delay')

        self.update_target_network()

        self.memory = Uniform_Memory(buffer_size=100000)

        self.sum_q = 0
        self.actor_loss = 0
        self.critic_loss = 0

    def train(self, j):
        # sample from memory
        if self.batch_size < self.memory.get_count:
            mem = self.memory.sample(self.batch_size)
            s_rep = T.tensor(np.array([_[0] for _ in mem]), dtype=T.float).to(self.actor.device)
            a_rep = T.tensor(np.array([_[1] for _ in mem]), dtype=T.float).to(self.actor.device)
            r_rep = T.tensor(np.array([_[2] for _ in mem]), dtype=T.float).to(self.actor.device)
            s1_rep = T.tensor(np.array([_[3] for _ in mem]), dtype=T.float).to(self.actor.device)
            d_rep = T.tensor(np.array([_[4] for _ in mem]), dtype=T.float).to(self.actor.device)

            self.critic.eval()
            self.actor.eval()
            self.critic_delay.eval()
            self.actor_target.eval()
            self.critic_target.eval()
            self.critic_target_delay.eval()

            # Calculate critic and train
            # s1_rep = self.preprocess_image(s1_rep)
            targ_actions = self.actor_target.forward(s1_rep)
            # targ_actions = targ_actions + T.clamp(T.Tensor(np.random.normal(scale=.2)), -.5, .5)

            target_q = self.critic_target.forward(s1_rep, targ_actions)
            target_q_delay = self.critic_target_delay.forward(s1_rep, targ_actions)

            q = self.critic.forward(s_rep, a_rep)
            q_delay = self.critic_delay.forward(s_rep, a_rep)

            target_q = target_q.view(-1)
            target_q_delay = target_q_delay.view(-1)

            target_q_min = T.min(target_q, target_q_delay)

            y_i = r_rep + self.GAMMA * target_q_min * (1-d_rep)
            y_i = y_i.view(self.batch_size, 1)

            self.critic.train()
            self.critic_delay.train()
            self.critic.optimizer.zero_grad()
            self.critic_delay.optimizer.zero_grad()

            critic_loss = T.nn.functional.mse_loss(y_i, q)
            # critic_loss_delay = T.nn.functional.mse_loss(y_i, q_delay)
            critic_loss.backward()
            self.critic.optimizer.step()
            self.critic_delay.optimizer.step()
            self.critic.eval()
            self.critic_delay.eval()

            # Calculate actor and train
            if j % self.policy_update_delay == 0:
                self.actor.optimizer.zero_grad()
                actions = self.actor.forward(s_rep)
                self.actor.train()
                actor_loss = T.mean(-self.critic.forward(s_rep, actions))
                actor_loss.backward()
                self.actor.optimizer.step()
                self.actor.eval()

            # update target networks
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
        

        critic_delay = self.critic_delay.named_parameters()
        critic_targ_delay_params = self.critic_target_delay.named_parameters()
        critic_delay_dict = dict(critic_delay)
        critic_targ_delay_dict = dict(critic_targ_delay_params)
        for name in critic_delay_dict:
            critic_delay_dict[name] = tau*critic_delay_dict[name].clone() +\
                                (1-tau)*critic_targ_delay_dict[name].clone()


        self.actor_target.load_state_dict(actor_dict)
        self.critic_target.load_state_dict(critic_dict)
        self.critic_target_delay.load_state_dict(critic_delay_dict)

    def action(self, state):
        self.actor.eval()
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        act = self.actor.forward(state).to(self.actor.device)
        self.actor.train()
        return act.cpu().detach().numpy()[0]

    def preprocess_image(self,image):
        # pytorch image: C x H x W
        image_swp = np.swapaxes(image, -1, 0)
        image_swp = np.swapaxes(image_swp,-1, -2)
        return image_swp/255.0

    def load_model(self):
        self.actor.load_state_dict(T.load(os.path.join(self.save_dir, self.actor.chkpt)))
        self.critic.load_state_dict(T.load(os.path.join(self.save_dir, self.critic.chkpt)))
        self.actor_target.load_state_dict(T.load(os.path.join(self.save_dir, self.actor_target.chkpt)))
        self.critic_target.load_state_dict(T.load(os.path.join(self.save_dir, self.critic_target.chkpt)))

    def save_model(self):
        self.actor.save_model()
        self.critic.save_model()
