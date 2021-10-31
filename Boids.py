import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation

from scipy.spatial.distance import squareform, pdist



class Boids():

    def __init__(self, 
               n_agents=30,
               box_size=1,
               max_speed=0.018,
               min_speed=0.01,
               max_acc=0.001,
               min_acc=0.0,
               separation_radius=0.1,
               alignment_radius=0.2,
               cohesion_radius=0.2,
               radius_mode=True,
               n_nearest=5,
               separation_factor=1,
               alignment_factor=1,
               cohesion_factor=1,
               random_agents=None
               ):

        self.radius_mode = radius_mode
        self.n_agents = n_agents
        self.n_nearest = n_nearest
        self.max_acc = max_acc
        self.min_acc = min_acc
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.pos = np.random.rand(self.n_agents, 2)
        self.angles = 2 * np.pi * np.random.rand(self.n_agents)
        self.vel = (np.multiply(np.vstack([np.cos(self.angles), 
                                           np.sin(self.angles)]).transpose(),
                                np.random.uniform(0.012, 0.018, (self.n_agents, 2))))
        self.box_size = box_size
        self.separation_radius = separation_radius
        self.cohesion_radius = cohesion_radius
        self.alignment_radius = alignment_radius
        self.separation_factor = separation_factor
        self.alignment_factor = alignment_factor
        self.cohesion_factor = cohesion_factor
        self.random_agents = random_agents



    def periodic_boundaries(self):
        self.pos = self.pos % self.box_size

    def update_Boids(self):
        # Perform single simulation step
        
        self.calc_distances()
        
        acceleration = self.acceleration()
        if self.random_agents is not None:
            acceleration[self.random_agents] = (
                np.random.uniform(-1, 1, 
                                  (len(self.random_agents), 2)) * self.max_acc)
            
        self.vel += acceleration
        self.vel = self.limit_vectors(self.vel, self.max_speed, self.min_speed)
        self.pos += self.vel
        self.angles = np.arctan2(self.vel[:,1], self.vel[:,0])

        self.periodic_boundaries()

        return self.pos, self.vel, self.angles

    def norm_vectors(self, input_matrix):
        normed = input_matrix / np.linalg.norm(input_matrix, axis=1).reshape(self.n_agents, -1)
        return normed

    def limit_vectors(self, input_matrix, u_limit, l_limit):
        limited = input_matrix.copy()
        normed = np.linalg.norm(input_matrix, axis=1).reshape(self.n_agents, -1)
        u_limit_factor = np.where(normed <= u_limit, 1, 1 / normed * u_limit)
        l_limit_factor = np.where(normed >= l_limit, 1, 1 / normed * l_limit)
        limited *= u_limit_factor * l_limit_factor
        limited = limited.reshape(self.n_agents, -1)

        return limited

    """
    distances_x = pdist(data[0].reshape(-1,1))
    distances_y = pdist(data[1].reshape(-1,1))
    distances_x[distances_x > 0.5 * periodic_l] -= periodic_l
    distances_y[distances_y > 0.5 * periodic_l] -= periodic_l

    distances = np.sqrt(distances_x**2 + distances_y**2)
    square_mat = squareform(distances)
    """




    def calc_distances(self):
        distances_x = pdist(self.pos[:, 0].reshape(-1,1))
        distances_y = pdist(self.pos[:, 1].reshape(-1,1))
        distances_x[distances_x > 0.5 * self.box_size] -= self.box_size
        distances_y[distances_y > 0.5 * self.box_size] -= self.box_size
        distances = np.sqrt(distances_x**2 + distances_y**2)
        self.distMatrix = squareform(distances)

        if self.radius_mode:
            self.distMatrix_sep = (0 < self.distMatrix) & (self.distMatrix < self.separation_radius)
            self.distMatrix_algn = (0 < self.distMatrix) & (self.distMatrix < self.alignment_radius)
            self.distMatrix_coh = (0 < self.distMatrix) & (self.distMatrix < self.cohesion_radius)

        else:
            output_matrix = np.zeros((self.n_agents, self.n_agents), dtype=bool)

            distance_sorted = np.argsort(self.distMatrix)
            nearest_idx = distance_sorted[:, 1 : self.n_nearest + 1]
            row_idx = np.multiply(np.arange(self.n_agents).reshape(self.n_agents, -1),
                                  np.ones((self.n_agents, self.n_nearest))).astype(int)

            output_matrix[row_idx, nearest_idx] = True
            self.distMatrix_sep = output_matrix #& (0 < self.distMatrix) & (self.distMatrix < self.separation_radius)
            self.distMatrix_algn = output_matrix #& (0 < self.distMatrix) & (self.distMatrix < self.alignment_radius)
            self.distMatrix_coh = output_matrix #& (0 < self.distMatrix) & (self.distMatrix < self.cohesion_radius)


    def acceleration(self):
        # Calculate total acceleration
        
        resulting_acc = 0
        resulting_acc += self.separation_factor * self.separation()
        resulting_acc += self.alignment_factor * self.alignment()
        resulting_acc += self.cohesion_factor * self.cohesion()
        resulting_acc /= (self.separation_factor + self.alignment_factor + self.cohesion_factor)

        return self.limit_vectors(resulting_acc, self.max_acc, self.min_acc)

    def separation(self):
        # apply rule #1 - Separation
        
        Distance_mask = self.distMatrix_sep
        sep_steer = self.pos * Distance_mask.sum(axis=1).reshape(self.n_agents, 1) - Distance_mask.dot(self.pos)
        sep_steer /= np.where(Distance_mask.sum(axis=1) == 0, 1, Distance_mask.sum(axis=1)).reshape(self.n_agents, -1)
        sep_steer -= self.vel
        
        return sep_steer    #self.norm_vectors(sep_steer)

    def alignment(self):
        # apply rule #2 - Alignment
        
        Distance_mask = self.distMatrix_algn
        algn_steer = Distance_mask.dot(self.vel)
        algn_steer /= np.where(Distance_mask.sum(axis=1) == 0, 1, Distance_mask.sum(axis=1)).reshape(self.n_agents, -1)
        algn_steer -= self.vel
        
        return algn_steer   #self.norm_vectors(algn_steer)

    def cohesion(self):
        # apply rule #1 - Cohesion
        
        Distance_mask = self.distMatrix_coh
        coh_steer = Distance_mask.dot(self.pos)
        coh_steer /= np.where(Distance_mask.sum(axis=1) == 0, 1, Distance_mask.sum(axis=1)).reshape(self.n_agents, -1)
        coh_steer -= self.pos
        coh_steer -= self.vel
        return coh_steer    #self.norm_vectors(coh_steer)
