import numpy as np
import torch as T
from replay_memory import Per_Memory, Uniform_Memory
import os
from model import Actor, Critic

class DDPG():
    def __init__(self,n_states, n_action, action_bound, layer_1_nodes, layer_2_nodes, actor_lr, critic_lr, PER, GAMMA,
                 tau, batch_size, save_dir):

        self.GAMMA = GAMMA
        self.batch_size = batch_size
        self.tau = tau
        self.PER = PER

        self.save_dir = save_dir

        self.actor = Actor(n_action, action_bound, layer_1_nodes, layer_2_nodes)
        self.critic = Critic(layer_1_nodes, layer_2_nodes)

        self.actor_target = Actor(n_action, action_bound, layer_1_nodes, layer_2_nodes)
        self.critic_target = Critic(layer_1_nodes, layer_2_nodes)

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
            s_rep = T.tensor(np.reshape(np.array([_[0] for _ in mem]),(-1,96,96,3)), dtype=T.float)
            a_rep = T.tensor(np.array([_[1] for _ in mem]), dtype=T.float)
            r_rep = T.tensor(np.array([_[2] for _ in mem]), dtype=T.float)
            s1_rep = T.tensor(np.reshape(np.array([_[3] for _ in mem]),(-1,96,96,3)), dtype=T.float)
            d_rep = T.tensor(np.array([_[4] for _ in mem]), dtype=T.float)


            td_error, critic_loss = self.loss_critic(a_rep, d_rep, r_rep, s1_rep, s_rep)
            actor_loss = self.loss_actor(s_rep)

            if self.PER:
                for i in range(self.batch_size):
                    update_error = np.abs(np.array(T.reduce_mean(td_error)))
                    self.memory.update(idxs[i], update_error)

            self.sum_q += np.amax(T.squeeze(self.critic(s_rep, a_rep), 1))
            self.actor_loss += np.amax(actor_loss)
            self.critic_loss += np.amax(critic_loss)

            # update target network
            self.update_target_network(self.tau)

    def loss_actor(self, s_rep):
        self.actor.optimizer.zero_grad()
        actions = self.actor((s_rep))
        actor_loss = -T.mean(self.critic(s_rep, actions))
        actor_loss.backward()
        self.actor.optimizer.step()
        # actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        # self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        return actor_loss

    def loss_critic(self,a_rep, d_rep, r_rep, s1_rep, s_rep):
        
        targ_actions = self.actor_target(s1_rep)
        target_q = self.critic_target(s1_rep, targ_actions)
        y_i = r_rep + self.GAMMA * target_q * (1 - d_rep)

        self.ciritc.optimizer.zero_grad()
        q = self.critic(s_rep, a_rep)
        td_error = y_i - q
        if not self.PER:
            critic_loss = T.mean(T.square(td_error))
        else:
            critic_loss = T.mean(T.square(td_error) * self.isweight)
        
        ciritc_loss.backward()
        self.critic.optimizer.step()

        # critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        # self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))
        return td_error, critic_loss

    def preprocess(self, image):
        self.actor.eval()
        image = T.tensor([image], dtype=T.float)

        return image


    def update_target_network(self, tau=.001):
        actor = self.actor.named_parameters()
        actor_targ_params = self.actor_target.named_parameters()
        actor_dict = dict(actor)
        actor_targ_dict = dict(actor_targ_params)

        for name in actor_dict:
            actor_dict[name] = tau*actor_dict[name].clone() +\
                                (1-tau)*actor_targ_dict[name].clone()
        self.actor_target.load_state_dict(actor_dict)

        critic = self.critic.named_parameters()
        critic_targ_params = self.critic_target.named_parameters()
        critic_dict = dict(critic)
        critic_targ_dict = dict(critic_targ_params)

        for name in critic_dict:
            critic_dict[name] = tau*critic_dict[name].clone() +\
                                (1-tau)*critic_targ_dict[name].clone()
        self.critic_target.load_state_dict(critic_dict)


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
