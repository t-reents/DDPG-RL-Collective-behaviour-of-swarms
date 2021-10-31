# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 19:39:07 2021

@author: Timo
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
import datetime

from Agents import Agents
from Flock import Flock
from plot_func import plot


class Training():
    
    def __init__(self, environment_parameter, agents_parameter, n_episodes=120, max_steps=5000, record_step=5):
        self.env = Flock(**environment_parameter)
        agents_parameter.update(
            {"state_size": self.env.observation_space.shape[0]}
            )
        self.agent = Agents(**agents_parameter)
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.record_step = record_step
    
    def run(self):
        time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        folder_agents = "stored data/agents/{}/".format(time_now)
        folder_videos = "stored data/videos/{}/".format(time_now)
        os.chdir(".")
        if not os.path.isdir(folder_agents):
            os.makedirs(folder_agents, exist_ok=True)
        if not os.path.isdir(folder_videos):
            os.makedirs(folder_videos, exist_ok=True)
        file = open('stored data/agents/{}/environment.pckl'.format(time_now),'wb')
        pickle.dump(self.env, file)
        file.close()
        
        scores = []
        plots = []
        action_list = []
        actor_loss = []
        critic_loss = []
    
        for i_episode in tqdm(range(1,self.n_episodes)):
            states = self.env.reset()
            self.agent.reset()
            inner_scores = []
            inner_act = []
            
            for i_step in range(self.max_steps):
                actions = self.agent.act(states,add_noise=True)
                next_states, rewards, done, dist = self.env.step(actions)
                a_loss, c_loss = self.agent.step(states, actions, rewards, next_states, done)
                inner_scores.append(np.mean(rewards))
                states = next_states
                inner_act.append(actions)

            action_list.append(inner_act)
            scores.append(np.mean(inner_scores))
            print(f'mean score from last episode: {scores[-1]}')

            if i_episode % self.record_step == 0 and scores[-1] > -10.0:
                plots.append(plot(environment_input=self.env,
                                  agent_input=self.agent).get("animation"))
                plots[-1].save(folder_videos + 
                               f"{24:03}_{self.record_step*i_episode:03}.mp4")
                plt.close()
                self.agent.save_checkpoint(folder_agents +
                                           f'agents_trained_{i_episode}.tar', 
                                           i_episode)
                
        return action_list, scores, actor_loss, critic_loss

#%%

box = 1.0
n_agents = 30

env_parameter = {"n_agents": n_agents, "n_steps": 3000,
                 "flock_reward_scaling": 1,
                 "obstacle_penalty_scaling": 5, "vel_scaling": 1, 
                 "rotation_size": 0.011, "v_var_size": 0.00005,
                 "min_speed": 0.0125, "n_nearest": 5, "abs_max_speed": 0.03, 
                 "abs_min_speed": 0.01, "distant_threshold": 0.1, 
                 "proximity_threshold": 0.02, "box_size": box,
                 "max_speed": 0.025}

agents_parameter = {"action_size": 2, "num_agents": n_agents, 
                    "random_seed": 0}



train = Training(env_parameter, agents_parameter, record_step=1)

train.run()
