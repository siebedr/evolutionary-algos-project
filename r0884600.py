import Reporter
import numpy as np
import random
import time
from numba import njit
from plot import basic_plot
from math import sqrt, exp


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


@njit(fastmath=True)
def adapt_mutation(mutation_rate, lr):
    return min(0.7, mutation_rate * exp(lr * np.random.rand()))


@njit
def recombination_helper(path1, path2):
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

    return child


class Individual:
    def __init__(self, value, mutation_rate=0.5):
        self.value = value
        self.mutation_rate = mutation_rate

    def mutate(self, lr):
        self.mutation_rate = adapt_mutation(self.mutation_rate, lr)

        inverse_mutation(self.value, self.mutation_rate)

    def recombine(self, other):
        # Ordered crossover, returns child

        w = np.random.rand() - 0.5
        new_mutation_rate = self.mutation_rate - w*other.mutation_rate
        return Individual(recombination_helper(self.value, other.value), max(0.1, new_mutation_rate))


# NUMBA functions to improve speed
@njit
def cost_helper(path, dist):
    total = 0

    for i in range(len(path) - 1):
        val = path[i]
        next_val = path[i + 1]
        total += dist[val][next_val]

    return total + dist[path[len(path) - 1]][0]


class Population:
    def __init__(self, size, dist_matrix):
        self.size = size
        self.dist_matrix = dist_matrix

        self.individuals: list[Individual] = []
        self.init_population()

    def init_population(self):
        # Random initialization
        while len(self.individuals) < self.size:
            self.individuals.append(Individual(np.random.permutation(len(self.dist_matrix))))

    def elimination(self, offspring: list[Individual]):
        # Does elimination and replaces original population (alpha + mu)
        offspring.sort(key=lambda x: self.fitness(x), reverse=True)
        self.individuals = offspring[:self.size]

    def selection(self, k) -> Individual:
        # k tournament selection
        possible = random.choices(self.individuals, k=k)
        possible.sort(key=lambda x: self.fitness(x), reverse=True)
        return possible[0]

    def mutate_all(self, lr):
        for ind in self.individuals:
            ind.mutate(lr)

    def fitness(self, individual: Individual) -> float:
        return 1 / self.cost(individual)

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

    def __init__(self, distance_matrix, population_size, offspring_size, k, learning_rate):
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.k = k
        self.lr = learning_rate

        self.population = Population(self.population_size, self.distance_matrix)

    def step(self):
        offspring = []

        while len(offspring) < self.offspring_size:
            mother: Individual = self.population.selection(self.k)
            father: Individual = self.population.selection(self.k)
            child = mother.recombine(father)
            child.mutate(self.lr)
            offspring.append(child)

        self.population.mutate_all(self.lr)

        self.population.elimination(offspring)

        return self.population.best().value, self.population.best_fitness(), self.population.mean()


class r0884600:
    # PARAMETERS
    stop = 20
    population_size = 1000
    offspring_size = 1500
    k = 5
    learning_rate = None

    no_change = 0
    counter = 0

    meanObjective = 0.0
    bestObjective = 0.0
    bestSolution: Individual = None
    prev_obj = 0

    log_interval = 100

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def termination_on_best_converged(self):
        if self.prev_obj == self.bestObjective:
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

        self.learning_rate = 1 / sqrt(len(distanceMatrix))  # Proven to be optimal

        # Initialize the population.
        tsp = TSP(distanceMatrix, self.population_size, self.offspring_size, self.k, self.learning_rate)

        # Run the algorithm until termination condition is met.
        while self.termination_on_best_converged():
            self.counter += 1
            self.bestSolution, self.bestObjective, self.meanObjective = tsp.step()

            timeLeft = self.reporter.report(self.meanObjective, self.bestObjective, self.bestSolution)
            if timeLeft < 0:
                break

            if self.counter % self.log_interval == 0:
                print("\nIteration: ", self.counter, "\nMean: ", self.meanObjective, "\nBest: ", self.bestObjective,
                      "\nPath cost: ", 1 / self.bestObjective)

        print("\nIteration: ", self.counter, "\nMean: ", self.meanObjective, "\nBest: ", self.bestObjective,
              "\nPath cost: ", 1 / self.bestObjective)
        return 0


program = r0884600()
start = time.time()
program.optimize("./Data/tour250.csv")
end = time.time()
print("\nRUNTIME: ", end - start)
basic_plot()
