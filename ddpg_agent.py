
import numpy as np
import random

from collections import namedtuple, deque
from models import DDPG_Actor,DDPG_Critic
from utils import ExperienceReplay,PrioritizedExperienceReplay,NormalNoiseStrategy
# from prioritized_memory import Memory

import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    def __init__(self, state_size, action_size, num_agents, seed, \
                 gamma=0.99, tau=1e-3, lr_actor=1e-3, lr_critic=1e-2, \
                 buffer_size = 10e5, buffer_type = 'replay', policy_update = 1):
        # General info
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.t_step = 0
        self.gamma = gamma
        # Actor Network -- Policy-based
        self.actor = DDPG_Actor(state_size, action_size, hidden_dims=(128,128), seed = seed)
        self.target_actor = DDPG_Actor(state_size, action_size, hidden_dims=(128,128), seed = seed)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        # Critic Network -- Value-based
        self.critic = DDPG_Critic(state_size, action_size, hidden_dims=(128,128), seed = seed)
        self.target_critic = DDPG_Critic(state_size, action_size, hidden_dims=(128,128), seed = seed)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.tau = tau
        # Replay memory
        self.buffer_type = buffer_type
        self.memory = ExperienceReplay(action_size, int(buffer_size)) #ExperienceReplay
        self.per = PrioritizedExperienceReplay(capacity=int(buffer_size),alpha=0.6,beta=0.9,error_offset=0.001)
        # NormalNoiseStrategy
        self.normal_noise = NormalNoiseStrategy()
        # Delayed Updates from TD3
        self.policy_update = policy_update

    def select_action(self, state):
        return self.normal_noise.select_action(self.actor, state)

    def _critic_error(self, state, action, reward, next_state, done):
        done = int(done)
        reward = float(reward)
        with torch.no_grad():
            argmax_a = self.target_actor(next_state)
            q_target_next = self.target_critic(next_state, argmax_a)
            q_target = reward + (self.gamma*q_target_next*(1-done))
            q_expected = self.critic(state,action)
            td_error = q_expected - q_target.detach()
        return td_error.detach().numpy()

    def step(self, state, action, reward, next_state, done, batch_size = 64):
        self.t_step += 1
        if self.buffer_type == 'prioritized':
            if self.num_agents == 20:
                reward = np.asarray(reward)[:,np.newaxis]
                done = np.asarray(done)[:,np.newaxis]                
                for i in range(self.num_agents):
                    error = self._critic_error(state[i],action[i],reward[i],next_state[i],done[i])
                    self.per.add(error, (state[i], action[i], reward[i], next_state[i], done[i]))
            else:
                done = np.asarray(done)
                reward = np.asarray(reward)
                state = state.squeeze()
                next_state = next_state.squeeze()
                error = self._critic_error(state,action,reward,next_state,done)
                self.per.add(error,(state, action, reward, next_state, done))
                
            # train if enough samples
            if self.t_step > batch_size:
                experiences, mini_batch, idxs, is_weights = self.per.sample(batch_size)
                self.learn(experiences,batch_size,idxs,is_weights)
        
        # add to replay buffer
        else:
            if self.num_agents == 20:
                reward = np.asarray(reward)[:,np.newaxis]
                done = np.asarray(done)[:,np.newaxis]
                for i in range(self.num_agents):
                    self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])
            else:
                self.memory.add(state, action, reward, next_state, done)
            # train if enough samples
            if len(self.memory) > batch_size:
                experiences = self.memory.sample(batch_size)
                self.learn(experiences,batch_size)

    def learn(self,experiences,batch_size,idxs=0,is_weights=0):
        states, actions, rewards, next_states, dones = experiences

        # *** 1. UPDATE Online Critic Network ***
        # 1.1. Calculate Targets for Critic
        argmax_a = self.target_actor(next_states)
        q_target_next = self.target_critic(next_states, argmax_a)
        q_target = rewards + (self.gamma*q_target_next*(1-dones))
        q_expected = self.critic(states,actions)
        # 1.2. Compute loss
        td_error = q_expected - q_target.detach()
        
        if self.buffer_type == 'prioritized':
            # PER --> update priority
            with torch.no_grad():
                error = td_error.detach().numpy()
                for i in range(batch_size):
                    idx = idxs[i]
                    self.per.update(idx, error[i])
            value_loss = (torch.FloatTensor(is_weights) * td_error.pow(2).mul(0.5)).mean()
        else:
            value_loss = td_error.pow(2).mul(0.5).mean()
            # value_loss = F.mse_loss(q_expected,q_target)
        # 1.3. Update Critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        if self.t_step % self.policy_update == 0:
            """
                Delaying Target Networks and Policy Updates from:
                ***Addressing Function Approximation Error in Actor-Critic Methods***
            """
            # *** 2. UPDATE Online Actor Network ***
            argmax_a = self.actor(states)
            max_val = self.critic(states,argmax_a)
            policy_loss = -max_val.mean() # add minus because its gradient ascent
            # Update Actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optimizer.step()
    
            # 3. UPDATE TARGET networks
            self.soft_update(self.actor,self.target_actor,self.tau)
            self.soft_update(self.critic,self.target_critic,self.tau)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
