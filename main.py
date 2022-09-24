import math
import random
import numpy as np


class MarsMap:
    def __init__(self):
        self.initial_x_range = [-170, -90]
        self.initial_y_range = [-20, 40]
        self.num_obstacles = 4
        self.obstacle_coords = [(-133, 18), (-121, -9), (-113, 1), (-104, 12)]

    # returns a list of the distances of (x, y) from the mountains
    def get_distance_from_obstacles(self, x, y):
        distances = []
        for i in range(self.num_obstacles):
            distance = math.sqrt(math.pow(x-self.obstacle_coords[i][0], 2) + math.pow(y-self.obstacle_coords[i][1], 2))
            distances.append(distance)
        return distances


class ParticleFilter:
    def __init__(self, map):
        self.n_samples = 1000
        self.resampling_fraction = 0.5
        self.map = map
        self.particles = self.initialize_particles()
        self.weights = [-1] * self.n_samples

    # initializes the particle coordinates
    def initialize_particles(self):
        particles = []
        for i in range(self.n_samples):
            x = random.uniform(self.map.initial_x_range[0], self.map.initial_x_range[1])
            y = random.uniform(self.map.initial_y_range[0], self.map.initial_y_range[1])
            particles.append([x, y])
        return particles

    # returns the final coordinate of the explorer
    def predict_location(self, distances):
        for i in range(len(distances)):
            self.update_particles()
            self.get_weights(distances[i])
            self.resample()
        location = [0, 0]
        self.weights = sorted(self.weights, reverse = True)
        self.particles = sorted(self.particles, reverse = True)
        for i in range(int(self.n_samples/4)):
            location[0] += self.weights[i] * self.particles[i][0]
            location[1] += self.weights[i] * self.particles[i][1]
        location[0] /= self.n_samples/4
        location[1] /= self.n_samples/4
        return location

    # P(dx1) is from Gaussian(2,1)
    # P(dy1) is from Gaussian(1,1)
    # (x1,y1) = (x0,y0) + (dx1,dy1)
    def update_particles(self):
        dx_list = self.sample_from_gaussian(2, 1, self.n_samples)
        dy_list = self.sample_from_gaussian(1, 1, self.n_samples)
        for i in range(self.n_samples):
            self.particles[i][0] += dx_list[i]
            self.particles[i][1] += dy_list[i]
        return

    # weight = Product of Gaussian_distribution(estimated_dist_i - actual_dist_i) for all i
    def get_weights(self, distance_dict):
        for i in range(self.n_samples):
            actual_distances = self.map.get_distance_from_obstacles(self.particles[i][0], self.particles[i][1])
            weight = 1
            for j in range(4):
                x = distance_dict[j] - actual_distances[j]
                weight *= self.gaussian_pdf(x, 2, 1)
            self.weights[i] = weight
        return

    # a resample_fraction of the particles are removed and resamples around the remaining particles
    def resample(self):
        n_resample = int(self.n_samples * self.resampling_fraction)
        sorted_weights_particles = [list(x) for x in zip(*sorted(zip(self.weights, self.particles), key=lambda pair: pair[0]))]
        self.weights = sorted_weights_particles[0][n_resample:self.n_samples]
        self.particles = sorted_weights_particles[1][n_resample:self.n_samples]
        dx_list = self.sample_from_gaussian(2, 1, self.n_samples - n_resample)
        dy_list = self.sample_from_gaussian(1, 1, self.n_samples - n_resample)
        sampled_indices = self.sample_from_weights(self.n_samples - n_resample)
        for i in range(self.n_samples - n_resample):
            self.particles.append([self.particles[sampled_indices[i]][0] + dx_list[i], self.particles[sampled_indices[i]][1] + dy_list[i]])
            self.weights.append(1)
        return

    # returns a list of n indices samples from the probabilities of weights, which are already sorted in ascending order
    def sample_from_weights(self, n):
        cumulative = []
        sum = 0
        for i in range(len(self.weights)):
            sum += self.weights[i]
            cumulative.append(sum)
        indices = []
        for i in range(n):
            uniform = random.uniform(0, cumulative[len(cumulative) - 1])
            for j in range(len(cumulative)):
                if cumulative[j] > uniform:
                    indices.append(j)
                    break
        return indices

    # returns a list of n samples from a Gaussian distribution with mean and variance equal to mean and var
    def sample_from_gaussian(self, mean, var, n):
        samples = []
        for i in range(n):
            uniform_a, uniform_b = np.random.uniform(0, 1, 2)
            sample = math.sqrt(var * -2 * math.log(uniform_a)) * math.cos(2 * math.pi * uniform_b) + mean
            samples.append(sample)
        return np.array(samples)

    # returns the probability of x using gaussian pdf
    def gaussian_pdf(self, x, mean, var):
        y = 1 / math.sqrt(2 * math.pi * var) * math.exp(-0.5 * math.pow((x - mean), 2) / var)
        return y


if __name__ == '__main__':
    distances = []
    num_dist = 20
    tmp_distances = [[],[],[],[]]
    # read distances
    for i in range(4):
        input()
        for k in range(num_dist):
            tmp_distances[i].append(float(input()))
    for i in range(num_dist):
        distances.append([tmp_distances[0][i],tmp_distances[1][i],tmp_distances[2][i],tmp_distances[3][i]])
    mars_map = MarsMap()
    filter = ParticleFilter(mars_map)
    x, y = filter.predict_location(distances)
    print(int(np.ceil(x/10) * 10))
    print(int(np.ceil(y/10) * 10))
    