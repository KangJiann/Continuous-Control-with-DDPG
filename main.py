#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alain
"""

from ddpg_agent import Agent
from unityagents import UnityEnvironment

import numpy as np
import pickle
import torch
from collections import deque

###############################################################################
# hyperparams
EPISODES = 1
GAMMA = .99
TAU = 1e-3
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
BUFFER_SIZE = int(1e6)
BUFFER_TYPE = 'replay'#'prioritized'
BATCH_SIZE = 64
POLICY_UPDATE = 2 # for normal updates =1
SEED = 0
# other required params
# path = 'Reacher_Linux_20agents/Reacher.x86_64'
path = 'Reacher_Linux_20agents/Reacher.x86_64'
algorithm = 'DDPG'
mode = 'evaluation'
results_filename = 'scores/twenty_agents/scores_' + algorithm + '_batch64_' + mode
###############################################################################

def defineEnvironment(path,verbose=False):
    # set the path to match the location of the Unity environment
    env = UnityEnvironment(file_name=path, worker_id= np.random.randint(0,int(10e6)))
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    if verbose:
        print('Number of agents:', len(env_info.agents))
        print('Number of actions:', action_size)
        print('States have length:', state_size)
    return env, brain_name, state_size, action_size, len(env_info.agents)

def playRandomAgent(env,brain_name,action_size=4,num_agents=1):
    env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

def train_agent(agent,env,brain_name,n_episodes=300, batch_size = BATCH_SIZE, filename='model_weights.pth'):
    scores = []
    scores_window = deque(maxlen=100)
    print_every = 1
    stop_criteria = 30 #+30 (over 100 consecutive episodes)
    aux = True
    checkpoint = 0
    # ------------------- begin training ------------------- #
    for e in range(1,n_episodes+1):
        # --- New Episode --- #
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get the current state
        state = env_info.vector_observations
        score = 0
        # --- Generate trajectories --- #
        while True:
            # get value of the 4 continuous actions
            action = agent.select_action(state)
            # get reward & next_states
            env_info = env.step(action)[brain_name]           # send all actions to the environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished
            # record trajectory
            agent.step(state,action,reward,next_state,done, batch_size)
            # update score
            score += np.mean(env_info.rewards)
            #check if finished
            if np.any(done):
                break
            else:
                state = next_state

        # Update monitorization variables & params for next Episode
        scores.append(score)
        scores_window.append(score)
        if e % print_every == 0:
            print('Episode {}/{}\tCurrent score: {:.2f}\tAvg Score: {:.2f}'.format(e,n_episodes,score,np.mean(scores_window)))
        if np.mean(scores_window) >= stop_criteria and aux:
            print('Environment solved in {} episodes'.format(e))
            checkpoint = e
            aux = False

    # save the model weights
    torch.save(agent.actor.state_dict(), 'weights/twenty_agents/actor_batch64_'+ filename)
    torch.save(agent.critic.state_dict(), 'weights/twenty_agents/crtic_batch64_'+ filename)
    return scores,checkpoint

def evaluate_agent(agent,env,brain_name,n_episodes=100):
    scores = []
    # ------------------- begin training ------------------- #
    for e in range(1,n_episodes+1):
        # --- New Episode --- #
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get the current state
        state = env_info.vector_observations
        score = 0
        # --- Visits --- #
        while True:
            # Agent selects an action
            action = agent.select_action_evaluation(state)
            # get reward & next_states
            env_info = env.step(action)[brain_name]           # send all actions to the environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished
            # Update monitorization variables & params for next visit
            score += np.mean(env_info.rewards)
            if np.any(done):
                break
            else:
                state = next_state
        # Update monitorization variables & params for next Episode
        scores.append(score)
        print('Episode/Test {} throws an avg of {}'.format(e,score))
    return scores

if __name__ == "__main__":
    # set environment and get state & action size
    env, brain_name, state_size,action_size, num_agents = defineEnvironment(path,verbose=True)

    # define agent
    agent = Agent(state_size,action_size,num_agents, \
                  SEED,GAMMA,TAU,LR_ACTOR,LR_CRITIC, \
                      BUFFER_SIZE, BUFFER_TYPE, POLICY_UPDATE)
    if mode == 'train':
        # train
        scores, checkpoint = train_agent(agent,env,brain_name,n_episodes=EPISODES, batch_size = BATCH_SIZE)
        # export data
        with open(results_filename,'wb') as f:
            pickle.dump([scores,checkpoint],f)
    elif mode == 'evaluation':
        weights_filename = 'weights/twenty_agents/actor_batch64_model_weights.pth'
        agent.actor.load_state_dict(torch.load(weights_filename))
        agent.actor.eval()
        scores = evaluate_agent(agent,env,brain_name,n_episodes=EPISODES)
        checkpoint = None
    # close env
    env.close()
        