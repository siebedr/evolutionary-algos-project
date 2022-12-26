import random
import time

import numpy as np
from numba import njit

import Reporter
from plot import basic_plot, plot_mutation_rate


# NUMBA functions to improve speed
@njit
def swap_mutation(path, mutation_rate):
    if np.random.rand() < mutation_rate:
        new_index1 = np.random.randint(len(path))
        new_index2 = np.random.randint(len(path))

        swap = path[new_index1]
        path[new_index1] = path[new_index2]
        path[new_index2] = swap


@njit
def inverse_mutation(path, mutation_rate):
    if np.random.rand() < mutation_rate:
        rand1 = np.random.randint(len(path))
        rand2 = np.random.randint(len(path))

        start_node = min(rand1, rand2)
        end_node = max(rand1, rand2)

        path[start_node:end_node] = path[start_node:end_node][::-1]
        return start_node, end_node
    return 0, 0


@njit
def ordered_crossover(path1, path2, mut_1, mut_2):
    rand1 = np.random.randint(len(path1))
    rand2 = np.random.randint(len(path1))

    start_node = min(rand1, rand2)
    end_node = max(rand1, rand2)

    other_values = [x for x in path2 if x not in path1[start_node:end_node]]

    child = np.empty(len(path1), dtype=np.int32)

    for i in range(start_node):
        child[i] = other_values.pop(0)

    for i in range(start_node, end_node):
        child[i] = path1[i]

    for i in range(end_node, len(path1)):
        child[i] = other_values.pop(0)

    # Calculate combined mutation rate
    beta = 2 * np.random.rand() - 0.5
    new_rate = max(0.01, mut_1 + beta * (mut_2 - mut_1))

    return child, new_rate


@njit
def PMX_crossover(path1, path2, mut_1, mut_2):
    rand1 = np.random.randint(len(path1))
    rand2 = np.random.randint(len(path1))

    start_node = min(rand1, rand2)
    end_node = max(rand1, rand2)

    child = np.empty(len(path1), dtype=np.int32)

    # Copy swath
    child[start_node:end_node] = path1[start_node:end_node]

    # Values of p2 in swath that are not in child
    other_values = [x for x in path2[start_node:end_node] if x not in path1[start_node:end_node]]

    for i in range(len(path1)):
        if i < start_node or i >= end_node:
            if path2[i] not in path1[start_node:end_node]:
                child[i] = path2[i]
            else:
                child[i] = other_values.pop(0)

    # Calculate combined mutation rate
    beta = 2 * np.random.rand() - 0.5
    new_rate = max(0.01, mut_1 + beta * (mut_2 - mut_1))

    return child, new_rate


@njit
def cycle_crossover(path1, path2, mut_1, mut_2):
    child = np.full(len(path1), -1, dtype=np.int32)
    index = np.random.randint(len(path1))
    node = path1[index]

    # Cycle from p1
    while node not in child:
        child[index] = node
        index = np.where(path1 == path2[index])[0][0]
        node = path1[index]

    # Fill with remaining values from p2
    for i in range(len(path1)):
        if child[i] == -1:
            child[i] = path2[i]

    # Calculate combined mutation rate
    beta = 2 * np.random.rand() - 0.5
    new_rate = max(0.01, mut_1 + beta * (mut_2 - mut_1))

    return child, new_rate


@njit
def two_opt(path, dist, depth=-1):
    # Pre calculate distances
    dists = np.zeros(len(path))
    reverse_dists = np.zeros(len(path))

    for i in range(len(path)):
        dists[i] = dist[path[i], path[(i + 1) % len(path)]]
        reverse_dists[i] = dist[path[(i + 1) % len(path)], path[i]]

    best = path
    best_dist = np.sum(dists)
    counter = 0

    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            new_cost = np.sum(dists[:i])
            new_cost += np.sum(reverse_dists[i:j + 1])
            new_cost += np.sum(dists[j + 1:])
            new_cost += dist[path[i], path[(i + 1) % len(path)]] + dist[path[j], path[(j + 1) % len(path)]]

            if new_cost < best_dist:
                counter += 1
                best = np.concatenate((path[:i], path[i:j + 1][::-1], path[j + 1:]))
                best_dist = new_cost

            if depth != -1 and counter > depth:
                return best

    return best


@njit
def light_two_opt(path, dist):
    best = path
    best_dist = cost_helper(path, dist)

    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            best = np.concatenate((path[:i], path[i:j + 1][::-1], path[j + 1:]))
            new_cost = cost_helper(best, dist)

            if new_cost < best_dist:
                return best
    return best


@njit
def compute_similarity(route1, route2):
    counter = 0

    for i in range(len(route1)):
        index = np.where(route2 == route1[i])[0][0]
        if route1[(i + 1) % len(route1)] == route2[(index + 1) % len(route2)]:
            counter += 1

    return counter / len(route1)


@njit
def hamming_distance(route1, route2):
    # result in percentage
    return (len(route1) - np.sum(route1 == route2)) / len(route1)


@njit
def create_route_dist_matrix(routes):
    """Creates matrix that contains distances between routes in population"""
    matrix = np.zeros((len(routes), len(routes)), dtype=np.float32)

    # Count intersections
    for i in range(len(routes)):
        for j in range(len(routes)):
            if j < i:
                matrix[i, j] = matrix[j, i]
            else:
                matrix[i, j] = hamming_distance(routes[i], routes[j])
    return matrix


@njit
def fitness_sharing_with_matrix(path_ind, route_dist_matrix, fitness, sigma=0.2, alpha=1):
    return fitness_sharing_with_list(route_dist_matrix[path_ind, :], fitness, sigma, alpha)


@njit
def fitness_sharing_with_list(route_distances, fitness, sigma=0.2, alpha=1):
    beta = 0

    for i in range(len(route_distances)):
        dist = route_distances[i]
        if dist <= sigma:
            beta += 1 - (dist / sigma) ** alpha

    result = fitness / beta ** np.sign(fitness)
    return result


@njit
def cost_helper(path, dist):
    total = 0

    for i in range(len(path)):
        val = path[i]
        next_val = path[(i + 1) % len(path)]
        total += dist[val][next_val]

    return total


@njit
def calc_route_distances(indices: set, routes, route):
    route_distances = np.ones((len(routes)), dtype=np.float32)

    for i in indices:
        route_distances[i] = hamming_distance(route, routes[i])

    return route_distances


@njit
def add_survivor(i, fitness, survivors, battling, surviving_route_distances, population):
    survivors[i] = np.argmax(fitness)
    battling.remove(survivors[i])
    fitness[survivors[i]] = -1
    best_route_distances = calc_route_distances(battling, population, population[survivors[i]])
    surviving_route_distances[i] = best_route_distances
    return [y for y, x in enumerate(best_route_distances) if x <= 0.2]


@njit
def shared_elimination(population, dist_matrix, keep, elites):
    """" Population should be alpha+mu """""

    survivors = np.empty(keep, dtype=np.int32)
    battling = set([i for i in range(len(population))])  # Indexes that are still in the game
    fitness = np.array([1 / cost_helper(x, dist_matrix) for x in population])

    relevant = set()  # Indexes of routes that are relevant to recalculate

    surviving_route_distances = np.empty((keep, len(population)), dtype=np.float32)

    for i in range(len(survivors)):
        # Update relevant fitness
        if i >= elites:
            for j in relevant:
                fitness[j] = fitness_sharing_with_list(surviving_route_distances[:i, j], fitness[j])

            relevant.clear()

        survivors[i] = np.argmax(fitness)
        battling.remove(survivors[i])
        fitness[survivors[i]] = -1
        best_route_distances = calc_route_distances(battling, population, population[survivors[i]])
        surviving_route_distances[i] = best_route_distances
        [relevant.add(y) for y, x in enumerate(best_route_distances) if x <= 0.2]

    return survivors


class Individual:
    def __init__(self, value, mutation_rate=None):
        self.value = value
        if mutation_rate is None:
            self.mutation_rate = 0.1 + (0.3 * np.random.rand())
            # self.mutation_rate = 0.6
        else:
            self.mutation_rate = mutation_rate
        self.neighbour_dist = None

    def mutate(self):
        inverse_mutation(self.value, self.mutation_rate)

    def local_search_operator(self, dist):
        self.value = light_two_opt(self.value, dist)

    def recombine(self, other):
        # Ordered crossover, returns child
        child, mut_rate = ordered_crossover(self.value, other.value, self.mutation_rate, other.mutation_rate)
        return Individual(child, mut_rate)


class Population:
    def __init__(self, size, dist_matrix, elites=0):
        self.route_dist_matrix = None
        self.size = size
        self.dist_matrix = dist_matrix

        self.individuals: list[Individual] = []
        self.init_population()
        self.elites = int(size * elites)

    def init_population(self):
        # Random initialization
        while len(self.individuals) < self.size:
            self.individuals.append(Individual(np.random.permutation(len(self.dist_matrix))))

    def elimination(self, offspring: list[Individual]):
        # Does elimination and replaces original population (alpha + mu)
        offspring += self.individuals[self.elites:]
        offspring.sort(key=lambda x: self.fitness(x), reverse=True)
        self.individuals = offspring[:(self.size-self.elites)] + self.individuals[:self.elites]
        self.individuals.sort(key=lambda x: self.fitness(x), reverse=True)

    def fs_elimination(self, offspring: list[Individual]):
        # Does elimination and replaces original population (alpha + mu)
        pop = self.individuals + offspring
        survivors = shared_elimination(np.array([i.value for i in pop]), self.dist_matrix, self.size, self.elites)

        self.individuals = [pop[x] for x in survivors]

    def selection(self, k) -> Individual:
        # k tournament selection
        possible = random.choices(self.individuals, k=k)
        possible.sort(key=lambda x: self.fitness(x), reverse=True)
        return possible[0]

    def update_route_dist_matrix(self):
        values = np.array([i.value for i in self.individuals])
        self.route_dist_matrix = create_route_dist_matrix(values)

    def mutate_all(self):
        for ind in self.individuals[self.elites:]:
            ind.mutate()

    # METRICS
    def fitness(self, individual: Individual) -> float:
        return 1 / self.cost(individual)

    def fitness_sharing(self, individual: Individual) -> float:
        return fitness_sharing_with_matrix(self.individuals.index(individual), self.route_dist_matrix,
                                           self.fitness(individual))

    def cost(self, individual: Individual) -> float:
        return cost_helper(individual.value, self.dist_matrix)

    def best(self) -> Individual:
        return self.individuals[0]

    def best_fitness(self) -> float:
        return self.fitness(self.individuals[0])

    def mean(self) -> float:
        total_fitness = 0
        for ind in self.individuals:
            total_fitness += self.fitness(ind)

        return total_fitness / self.size


class TSP:

    def __init__(self, distance_matrix, population_size, offspring_size, k, elites):
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.k = k

        self.population = Population(self.population_size, self.distance_matrix, elites)

    def ls_all(self, ls_individuals):
        for ind in ls_individuals:
            ind.local_search_operator(self.distance_matrix)

    def avg_mut_rate(self):
        total = 0
        for ind in self.population.individuals:
            total += ind.mutation_rate
        return total / self.population_size

    def step(self):
        offspring = []

        # Update route distance matrix for selection fitness sharing
        # self.population.update_route_dist_matrix()

        while len(offspring) < self.offspring_size:
            mother: Individual = self.population.selection(self.k)
            father: Individual = self.population.selection(self.k)
            child = mother.recombine(father)
            child.mutate()
            offspring.append(child)

        self.population.mutate_all()

        self.ls_all(self.population.individuals[:self.population.elites])

        self.population.fs_elimination(offspring)
        return self.population.best(), self.population.best_fitness(), self.population.mean()


class r0884600:
    # PARAMETERS
    stop = 100
    population_size = 50
    offspring_size = 100
    k = 5
    elites = 0.05

    no_change = 0
    counter = 0

    meanObjective = 0.0
    bestObjective = 0.0
    bestSolution: Individual = None
    prev_obj = 0

    log_interval = 50

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def termination_on_best_converged(self):
        if self.prev_obj >= self.bestObjective:
            self.no_change += 1
        else:
            self.no_change = 0
            self.prev_obj = self.bestObjective

        return self.no_change != self.stop

    def termination_on_mean_converged(self, decimals=8):
        if self.meanObjective - self.prev_obj < 10 ** -decimals:
            self.no_change += 1
        else:
            self.no_change = 0

        self.prev_obj = self.meanObjective

        return self.no_change != self.stop

    def termination_on_fixed_iterations(self, iteration):
        return self.counter != iteration

    # The evolutionary algorithm’s main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        distanceMatrix = np.loadtxt(filename, delimiter=",")

        # Initialize the population.
        tsp = TSP(distanceMatrix, self.population_size, self.offspring_size, self.k, self.elites)

        step_time = 0

        # Run the algorithm until termination condition is met.
        while self.termination_on_best_converged():
            step_start = time.time()
            self.counter += 1
            self.bestSolution, self.bestObjective, self.meanObjective = tsp.step()

            timeLeft = self.reporter.report(self.meanObjective, self.bestObjective, self.bestSolution.value,
                                            tsp.avg_mut_rate())
            if timeLeft < 0:
                break

            timing = time.time() - step_start
            step_time += timing

            if self.counter % self.log_interval == 0:
                print("\nIteration: ", self.counter, " (", timing, ")\nMean: ", self.meanObjective, "\nBest: ",
                      self.bestObjective,
                      "\nPath cost: ", cost_helper(self.bestSolution.value, distanceMatrix))

        print("\nIteration: ", self.counter, "\nMean: ", self.meanObjective, "\nBest: ", self.bestObjective,
              "\nPath cost: ", cost_helper(self.bestSolution.value, distanceMatrix), "\nAvg step time: ",
              step_time / self.counter, )
        return 0


program = r0884600()
start = time.time()
program.optimize("./Data/tour500.csv")
end = time.time()
print("\nRUNTIME: ", end - start)
basic_plot()
plot_mutation_rate()
