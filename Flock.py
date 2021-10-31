from typing import List, Tuple

import gym
import numpy as np
from numba import float32, int32, njit
from itertools import product as it_prod
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

# Need 32bit versions of π and 2π to keep types consistent
# inside numba functions
TPI = float32(2 * np.pi)
PI32 = float32(np.pi)


@njit(float32[:, :](float32[:], int32), nogil=True)
def _product_difference(a, n):
    """
    Generates 2d matrix of differences between all pairs in the argument array
    i.e. y[i][j] = x[j]-x[i]  for all i,j where i≠j

    Args:
        a (np.array): 1d array of 32bit floats
        n (int): Number of entries in array that will form 1st index of result

    Returns:
        np.array: 2d Array of differences
    """
    m = a.shape[0]
    d = np.empty((n, m - 1), dtype=float32)
    for i in range(n):
        for j in range(i):
            d[i][j] = a[j] - a[i]
        for j in range(i + 1, m):
            d[i][j - 1] = a[j] - a[i]
    return d


@njit(float32[:, :](float32[:], float32, int32), nogil=True)
def _shortest_vecs(a, length, m):
    """
    Get the shortest vector between pairs of points taking into account
    wrapping around the torus

    Args:
        a (np.array): 1d array of 32bit floats representing the points
        length (float): Length of the torus/closed loop
        m: (int): The width of the array

    Returns:
        np.array: 2d array of shortest vectors between all pairs of points
    """

    x = _product_difference(a, m)
    x_ = np.sign(x) * (np.abs(x) - length)
    return np.where(np.abs(x) < np.abs(x_), x, x_)


@njit(float32(float32, float32, float32), nogil=True)
def _shortest_vec(a, b, length):
    """
    Get the shortest vector between pairs of points taking into account
    wrapping around the torus

    Args:
        a (np.float32):
        b (np.float32):

    Returns:
        np.array: 2d array of shortest vectors between all pairs of points
    """
    x = a - b
    x_ = np.sign(x) * (np.abs(x) - length)
    return x if np.abs(x) < np.abs(x_) else x_


@njit(float32[:, :](float32[:, :], float32[:, :]), fastmath=True, nogil=True)
def _distances(xs, ys):
    """Convert x and y vector components to Euclidean distances"""
    return np.sqrt(np.power(xs, 2) + np.power(ys, 2))

@njit(
      float32[:](float32[:,:],float32[:],int32,int32,float32,float32, float32),
      fastmath=True, nogil=True
      )
def _new_velocity_reward(d,vel, num_agents, num_nearest, max_s, min_s, abs_max):
    vel_rewards = np.zeros(num_agents, dtype=float32)
    
    for i in range(num_agents):
        if min_s < vel[i] < max_s:
            vel_rewards[i] = np.exp(- 30000 * (vel[i] - 0.014) ** 2)
        else:
            vel_rewards[i] = -10

    return vel_rewards

def skew_reward(x):
    norm_x = np.linspace(0, 1, 10000)
    norm_dist = float32(skewnorm.pdf(norm_x, 60, loc=0.02, scale=0.025))

    return float32(skewnorm.pdf(x, 60, loc=0.02, scale=0.025) / max(norm_dist))

def linear(x, p1, p2):
    
    y = ((p2[1] - p1[1]) / (p2[0] - p1[0]) * x +
         (p1[1] * p2[0] - p2[1] * p1[0]) / (p2[0] - p1[0]))
    
    return y

def reward_v2_(x, lo, up):
    
    out = np.zeros(x.shape)
    
    x1 = x[x < lo]
    x2 = x[(lo <= x)]
    
    y1 = skew_reward(lo)
    
    out[x < lo] = linear(x1, (0, -5), (lo, y1))
    out[(lo <= x)] = skew_reward(x2)
    
    return out
   

def _new_distance_reward(d,proximity_threshold,distant_threshold,num_agents,num_nearest):
    
    dist_rewards = np.zeros((num_agents,num_nearest))
    for i in range(num_agents):
        sort_idx = np.argsort(d[i, : num_agents - 1])[:num_nearest]
        for j in range(num_nearest):
            n = sort_idx[j]
            dist_rewards[i, j] = d[i][n]
    dist_rewards = reward_v2_(dist_rewards, proximity_threshold, 
                              distant_threshold).reshape(num_agents, num_nearest)
    
    return dist_rewards.sum(axis = 1)


@njit(float32[:, :](float32[:]), fastmath=True, nogil=True)
def _relative_headings(theta):
    """
    Get smallest angle between heading of all pairs of

    Args:
        theta (np.array): 1d array of 32bit floats representing agent headings
            in radians

    Returns:
        np.array: 2d array of 32bit floats representing relative headings
            for pairs of boids
    """
    return _shortest_vecs(theta, TPI, theta.shape[0]) / PI32


@njit(float32(float32, float32), fastmath=True, nogil=True)
def _relative_heading(a, b):
    """
    Get smallest angle between heading of all pairs of

    Args:
        a (np.float32):
        b (np.float32):

    Returns:
        np.array: 2d array of 32bit floats representing relative headings
            for pairs of boids
    """
    return _shortest_vec(a, b, TPI) / PI32


@njit(
    (float32[:, :], float32[:], float32[:], int32, int32, int32, float32, float32,
     float32, float32), fastmath=True, nogil=True
)
def _observe(pos, theta, velocities, n_agents, n_obstacles, n_nearest,
             max_distance, box_size, max_speed, min_speed):
    """
    Returns a view on the flock phase space local to each agent. Since
    in this case all the agents move at the same speed we return the
    x and y components of vectors relative to each boid and the relative
    heading relative to each agent.

    In order for the agents to have similar observed states, for each agent
    neighbouring boids are sorted in distance order and then the closest
    neighbours included in the observation space

    Returns:
        np.array: Array of local observations for each agent, bounded to
            the range [-1,1]
    """
    xs = _shortest_vecs(pos[0], box_size, n_agents)
    ys = _shortest_vecs(pos[1], box_size, n_agents)
    d = _distances(xs, ys)

    obs_width = 4 * n_nearest + 2 + 2 * n_obstacles

    local_observation = np.zeros((n_agents, obs_width), dtype=float32)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    obs_x = xs[:, n_agents - 1 :]
    obs_y = ys[:, n_agents - 1 :]

    for i in range(n_agents):
        sort_idx = np.argsort(d[i, : n_agents - 1])[:n_nearest]
        cos_ti = cos_t[i]
        sin_ti = sin_t[i]
        xs_i = xs[i]
        ys_i = ys[i]
        obs_x_i = obs_x[i]
        obs_y_i = obs_y[i]
        theta_i = theta[i]
        for j in range(n_nearest):
            n = sort_idx[j]
            local_observation[i, j] = (
                cos_ti * xs_i[n] + sin_ti * ys_i[n]
            ) / max_distance
            local_observation[i, n_nearest + j] = (
                cos_ti * ys_i[n] - sin_ti * xs_i[n]
            ) / max_distance
            local_observation[i, 2 * n_nearest + j] = _relative_heading(
                theta_i, theta[n]
            )
            local_observation[i,3 * n_nearest +j] = (velocities[j] -
                                                     velocities[i]
                                                     )/max_speed
            
        local_observation[i,4 * n_nearest] = velocities[i]/max_speed
        local_observation[i,4 * n_nearest + 1] = min_speed/velocities[i]

        for k in range(n_obstacles):
            obs_x_ = (cos_ti * obs_x_i[k] + sin_ti * obs_y_i[k]) / max_distance
            obs_y_ = (cos_ti * obs_y_i[k] - sin_ti * obs_x_i[k]) / max_distance
            local_observation[i, 4 * n_nearest + 2 + k] = obs_x_
            local_observation[i, 4 * n_nearest + 2 + n_obstacles + k] = obs_y_

    return d, local_observation

class Flock(gym.Env):
    def __init__(
        self,
        n_agents: int,
        n_steps: int,
        flock_reward_scaling: float,
        obstacle_penalty_scaling: float,
        vel_scaling: float,
        rotation_size: float,
        v_var_size: int,
        min_speed: float,
        max_speed: float,
        abs_max_speed: float,
        abs_min_speed: float,
        distant_threshold: float,
        proximity_threshold: float,
        n_nearest: int,
        box_size: float ,
        v_distribution = None,
        initial_pos = None,
        initial_angle = None,
        obstacles: List[Tuple] = (),
    ):
        """
        Initialize a discrete action flock environment

        In this environment the boids are only allowed to rotate by a fixed
        amount at each step, the action space is then discrete values indexing
        these rotations

        Args:
            n_agents (int): Number of agents to include in simulation
            speed (float): Max allowed velocity of agents
            n_steps (int): Number of steps in an episode
            rotation_size (float): Smallest rotation size in radians
            n_actions (int): NUmber of allowed rotations actions, should be an
                odd integer >1
            distant_threshold (float): Distance cut-off for rewards
            proximity_threshold (float, optional): Distance at which other
                boids are considered too close for reward
            n_nearest (int): Number of agents to include in the local
                observations generated for each agent
        """
        assert (
            n_nearest <= n_agents
        ), "Number of agents in observation should be <= number of agents"
        assert distant_threshold > proximity_threshold

        self.n_agents = n_agents
        self.proximity_threshold = float32(proximity_threshold)
        self.distant_threshold = float32(distant_threshold)
        self.n_nearest = n_nearest
        self.rotation_size = rotation_size
        self.n_obstacles = len(obstacles)
        self.max_s = max_speed
        self.n_steps = n_steps
        self.flock_reward_scaling = flock_reward_scaling
        self.obstacle_penalty_scaling = obstacle_penalty_scaling


        self.x = np.zeros((2, n_agents + self.n_obstacles), dtype=np.float32)
        self.x[:, self.n_agents :] = np.array([i[:2] for i in obstacles]).T.astype(
            np.float32
        )

        self.speed = np.zeros(n_agents, dtype=np.float32)

        self.theta = np.zeros(n_agents, dtype=np.float32)

        self.obstacle_radii = np.array([i[2] for i in obstacles])[np.newaxis, :].astype(
            np.float32
        )

        observation_shape = (4 * n_nearest) + 2 + (2 * self.n_obstacles)

        self.observation_space = gym.spaces.box.Box(
            -1.0, 1.0, shape=(observation_shape,)
            )

        self.abs_v_ulim = abs_max_speed
        self.abs_v_llim = abs_min_speed
        self.vel_scaling = vel_scaling
        self.v_distribution = v_distribution
        self.v_var_size = v_var_size
        self.min_s = min_speed
        self.box_size = box_size
        self.initial_pos = initial_pos
        self.initial_angle = initial_angle
        self.max_distance = np.sqrt(2 * ((self.box_size / 2) ** 2))
        
    def _update_agents(self):
        """
        Update the position of all agents based on current
        speed and headings
        """
        act_vel = self.speed
        v0 = act_vel * np.cos(self.theta)
        v1 = act_vel * np.sin(self.theta)
        self.x[0][: self.n_agents] = (self.x[0][: self.n_agents] + v0) % self.box_size
        self.x[1][: self.n_agents] = (self.x[1][: self.n_agents] + v1) % self.box_size

    def _rotate_agents(self, actions: np.array):
        """
        Rotate the agents according to the argument actions indices
        Args:
            actions (np.array): Array of actions indexing the amount to
                rotate (steer) each of the agents by
        """

        actions_r = actions.transpose()[0]
        rot_actions = np.where(np.abs(actions_r) < 0.5, 0.5, np.abs(actions_r))
        rot_actions = (np.sign(actions_r) * (2 * rot_actions -1) 
                       * self.rotation_size * PI32)

        self.theta = float32(np.mod(self.theta + rot_actions, TPI))



    def _accelerate_agents(self, actions: np.array):
        """
        Accelerate agents.
        Args:
            actions (np.array): Array of actions to change the absolute
                                value of the velocity of each agent
        """

        actions_v = actions.transpose()[1]
        vel_actions = np.where(np.abs(actions_v) < 0.5, 0.5, np.abs(actions_v))
        vel_actions = float32(np.sign(actions_v) * (2 * vel_actions - 1)) * self.v_var_size

        self.speed = np.clip(self.speed + vel_actions,self.abs_v_llim,self.abs_v_ulim)

    def _obstacle_penalties(self, ds: np.array):
        """
        Return penalties for agent colliding with obstacles

        Args:
            ds (np.array): 2d array distances to obstacles for each agent

        Returns:

        """
        return np.any(ds < self.obstacle_radii, axis=1)

    def _rewards(self, d: np.array) -> np.array:
        """
        Get rewards for each agent based on distances to other boids

        Args:
            d (np.array): 2d array representing euclidean distances between
                each pair of boids

        Returns:
            np.array: 1d array of reward values for each agent
        """
        
        agent_rewards = self.flock_reward_scaling * _new_distance_reward(
            d[:, : self.n_agents - 1], self.proximity_threshold, self.distant_threshold,
            self.n_agents,self.n_nearest
        )
        obstacle_penalties = self.obstacle_penalty_scaling * self._obstacle_penalties(
            d[:, self.n_agents - 1 :]
        )
        velocity_rewards = self.vel_scaling * _new_velocity_reward(
            d[:, : self.n_agents - 1], self.speed, self.n_agents, self.n_nearest,
            self.max_s, self.min_s, self.abs_v_ulim
            )

        return (agent_rewards - obstacle_penalties + velocity_rewards)

    def _observe(self) -> np.array:
        """
        Returns a view on the flock phase space local to each agent. Since
        in this case all the agents move at the same speed we return the
        x and y components of vectors relative to each boid and the relative
        heading relative to each agent.

        In order for the agents to have similar observed states, for each agent
        neighbouring boids are sorted in distance order and then the closest
        neighbours included in the observation space

        Returns:
            np.array: Array of local observations for each agent, bounded to
                the range [-1,1]
        """
        return _observe(
            self.x,
            self.theta,
            self.speed,
            self.n_agents,
            self.n_obstacles,
            self.n_nearest,
            self.max_distance,
            self.box_size,
            self.abs_v_ulim,
            self.abs_v_llim
        )

    def step(self, actions: np.array) -> Tuple:
        """
        Step the model forward updating applying the steering actions to the
        agents, then updating the positions of the boids

        Args:
            actions (np.array): Array of steering actions applied to each agent
                actions index the array of discrete values

        Returns:
            tuple: Tuple in the format (local_observations, rewards, done, {})
                as per the open AI API
        """
        self._rotate_agents(actions)
        self._accelerate_agents(actions)
        self._update_agents()
        self.i += 1

        d, local_observations = self._observe()
        rewards = self._rewards(d)
        
        dones = ((self.speed == self.abs_v_llim) | 
                 (self.speed == self.abs_v_ulim))
        
        return local_observations, rewards, dones, d[:, : self.n_agents - 1]

    def reset(self) -> np.array:
        """
        Reset the environment assigning the agents random positions and
        headings but assigning them all the max allowed speed

        Returns:
            np.array: Array of local observations of the reset state
        """
        self.x[:, : self.n_agents] = np.random.random(size=(2, self.n_agents)).astype(
            np.float32 
        ) * self.box_size
        
        if self.initial_pos is not None:
            self.x[:, : self.n_agents] = np.copy(self.initial_pos)
        
        if self.v_distribution is None:
            self.v_distribution = np.ones(self.n_agents).astype(np.float32)*(self.max_s + self.min_s)/2
            self.speed = self.v_distribution
        else:
            if len(self.v_distribution) == self.n_agents:
                self.speed = np.copy(self.v_distribution)
            else:
                print('speed size is false')
        
        assert np.all(self.speed != 0), "velocity equal to 0"

        self.theta = TPI * np.random.random(self.n_agents).astype(np.float32)
        if self.initial_angle is not None:
            self.theta = np.copy(self.initial_angle)
        self.i = 0

        _, local_observations = self._observe()

        return local_observations
