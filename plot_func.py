# -*- coding: utf-8 -*-
"""
Created on Mon May 17 08:32:55 2021

@author: Timo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation


def plot_loghist(x, bins, density=None, logarithmic_x=False, logarithmic_y=False, axes=None):
    if logarithmic_x:
        _, bins = np.histogram(x, bins=bins)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.xscale('log')
    else:
        logbins = bins
    if axes:
        axes.hist(x, bins=logbins, log=logarithmic_y, density=density)
        return axes
    else:
        plt.hist(x, bins=logbins, log=logarithmic_y, density=density)
    




def plot(environment_input, agent_input, steps=2000, box_size=1.0):
    env = environment_input
    agent = agent_input
    state = env.reset()

    # Record positions, headings and rewards
    pos = []
    rot = []
    rwd = []
    vel = []
    act = []
    states = []
    distances = []
    vel.append(env.speed.copy())
    # Run the model taking actions from the RL agent
    for _ in range(steps):
        actions = agent.act(state,add_noise=False)
        state, reward, _, dist = env.step(actions)
        pos.append(env.x[:, :env.n_agents].copy())
        rot.append(env.theta.copy())
        rwd.append(reward)
        vel.append(env.speed.copy())
        distances.append(dist)
        act.append(actions)
        states.append(state)
    pos = np.stack(pos)
    rot = np.stack(rot)
    rwd = np.stack(rwd)
    distances = np.stack(distances)
    # Scale rewards to use as colours for the plot
    rwd = 255*(rwd-rwd.min())/(rwd.max()-rwd.min())
    
    d = np.append(pos, rot[:, np.newaxis, :], axis=1)
    d = np.append(d, rwd[:, np.newaxis, :], axis=1)
    d = np.append(d, np.min(distances, axis=2)[:, np.newaxis, :], axis=1)
    fig, ax = plt.subplots(1,1, figsize=(8, 8))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

    obstacles = env.x[:, -env.n_obstacles:].T
    radii = env.obstacle_radii[0]
    for o in zip(obstacles, radii):
        draw_circle = plt.Circle((o[0][0], o[0][1]),
                                 o[1],
                                 fill=True,
                                 alpha=0.2,
                                 color='r')
        ax.add_artist(draw_circle)
        
    cmap = matplotlib.colors.ListedColormap(['red', 'green', 'orange'])
    boundaries = [0, env.proximity_threshold, env.distant_threshold, env.max_distance]
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)


    q = ax.quiver(d[0][0], d[0][1],
                  np.cos(d[0][2]),
                  np.sin(d[0][2]),d[0][4],
                  cmap=cmap, norm=norm,
                  units='dots', width=3.5,
                  scale=0.05, headwidth=3, headlength=3,
                  headaxislength=3)
                  #cmap=plt.get_cmap('winter'))
    #ax.legend(fontsize=20,loc=1, bbox_to_anchor=(1.1, 0.99), prop={'size': 15})
    #plt.tight_layout()
    def update_quiver(f):
        """Updates the values of the quiver plot"""
        q.set_offsets(f[:2].T)
        q.set_UVC(np.cos(f[2]), np.sin(f[2]), f[4])
        return q,

    anim = animation.FuncAnimation(fig,
                                   update_quiver,
                                   frames=d[1:],
                                   interval=50,
                                   blit=False)
    plt.close()

    return {'animation':anim, 'positions':pos, 'velocities':vel,
            'orientations':rot, 'actions': act, 'states': states}
