import numpy as np
import torch
from model import Actor, Critic


class Memory():
    def __init__(self,batch_size):
        # init state, action, reward, state_, done
        self.state = []
        self.action = []
        self.reward = []
        self.val = []
        self.prob = []
        self.done = []
        self.batch_size = batch_size

    def get_memory(self):

        self.n_states = len(self.state)
        batch_st = np.arange(0, self.n_states, self.batch_size)
        idx = np.arange(self.n_states, dtype=np.int16)
        np.random.shuffle(idx)

        batches = [idx[i:i+self.batch_size] for i in batch_st]

        return np.array(self.state),\
            np.array(self.action),\
            np.array(self.reward),\
            np.array(self.val),\
            np.array(self.prob),\
            np.array(self.done),\
            batches
        
    def store_memory(self, state, action, reward, val, prob, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.val.append(val)
        self.prob.append(prob)
        self.done.append(done)

    def clear_memory(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.val.clear()
        self.prob.clear()
        self.done.clear()
        

class Agent():
    def __init__(self, num_state, num_action, ep=0.2, beta=3, c1=0.1, layer_1_nodes=512, layer_2_nodes=256, batch_size=64):
        
        self.ep = ep
        self.beta = beta
        self.c1 = c1
        self.gamma = .99
        self.g_lambda = 0.95

        self.actor = Actor(num_state, num_action, layer_1_nodes, layer_2_nodes)
        self.critic = Critic(num_state, layer_1_nodes, layer_2_nodes)
        self.memory = Memory(batch_size)

    def take_action(self,state):
        state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        
        prob_dist = self.actor(state)
        value = self.critic(state)
        action = prob_dist.sample()

        prob = torch.squeeze(prob_dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return prob, action, value

    def store_memory(self, state,action, prob, val, reward, done):
        self.memory.store_memory(state, action, reward, val, prob, done)

    def train(self):
        epochs = 5

        for epoch in range(epochs):
            state_mem, action_mem, reward_mem, val_mem, prob_mem, done_mem, batches = self.memory.get_memory()
            
            # Calcualte the advantage
            advan = np.zeros(len(reward_mem))

            for T in range(len(reward_mem)-1):
                a_t = 0
                discount = 1
                for k in range(T, len(reward_mem)-1):
                    a_t += discount * (reward_mem[k] + self.gamma * val_mem[k+1]*(1-done_mem[k]) \
                        - val_mem[k])
                    discount *= self.gamma * self.g_lambda
                advan[T] = a_t 
            advan = torch.tensor(advan).to(self.actor.device)
            values = torch.tensor(val_mem).to(self.actor.device)

            for batch in batches:
                states = torch.tensor(state_mem[batch], dtype=torch.float).to(self.actor.device)
                old_prob = torch.tensor(prob_mem[batch], dtype=torch.float).to(self.actor.device)
                actions = torch.tensor(action_mem[batch], dtype=torch.float).to(self.actor.device)
                
                # calculate r_t(theta)
                dist_new = self.actor(states)
                prob_new = dist_new.log_prob(actions)
                r_t = prob_new.exp() / old_prob.exp()        
                # L_clip
                prob_clip = torch.clamp(r_t, 1-self.ep, 1+self.ep) * advan[batch]
                weight_prob = advan[batch] * r_t
                actor_loss = -torch.min(weight_prob, prob_clip).mean()

                # critic loss
                V_t = torch.squeeze(self.critic(states))
                V_t1 = advan[batch] + values[batch]
                critic_loss = (V_t1 - V_t)**2
                critic_loss = critic_loss.mean()

                tot_loss = actor_loss + self.c1*critic_loss
                self.actor.optim.zero_grad()
                self.critic.optim.zero_grad()
                tot_loss.backward()
                self.actor.optim.step()
                self.critic.optim.step()

        self.memory.clear_memory()

        


