import Reporter
import numpy as np
import random
import time
from numba import njit
from plot import basic_plot


# NUMBA functions to improve speed
@njit
def swap_mutation(path):
    new_index1 = np.random.randint(len(path))
    new_index2 = np.random.randint(len(path))

    swap = path[new_index1]
    path[new_index1] = path[new_index2]
    path[new_index2] = swap


@njit
def inverse_mutation(path):
    rand1 = np.random.randint(len(path))
    rand2 = np.random.randint(len(path))

    start_node = min(rand1, rand2)
    end_node = max(rand1, rand2)

    path[start_node:end_node] = path[start_node:end_node][::-1]


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
    def __init__(self, value):
        self.value = value

    def mutate(self, prob):
        if np.random.random() < prob:
            inverse_mutation(self.value)

    def recombine(self, other):
        # Ordered crossover, returns child
        return Individual(recombination_helper(self.value, other.value))


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

    def mutate_all(self, prob):
        for ind in self.individuals:
            ind.mutate(prob)

    def fitness(self, individual: Individual) -> float:
        return 1 / self.cost(individual)

    def cost(self, individual: Individual) -> float:
        return cost_helper(individual.value, self.dist_matrix)

    def best(self) -> Individual:
        return self.individuals[0]

    def best_fitness(self) -> float:
        return self.fitness(self.individuals[0])

    def mean(self) -> Individual:
        total_fitness = 0
        for ind in self.individuals:
            total_fitness += self.fitness(ind)

        return total_fitness / self.size


class TSP:

    def __init__(self, distance_matrix, population_size, offspring_size, k, mutation_probability):
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.k = k
        self.mut_prob = mutation_probability

        self.population = Population(self.population_size, self.distance_matrix)

    def step(self):
        offspring = []

        while len(offspring) < self.offspring_size:
            mother: Individual = self.population.selection(self.k)
            father: Individual = self.population.selection(self.k)
            child = mother.recombine(father)
            child.mutate(self.mut_prob)
            offspring.append(child)

        self.population.mutate_all(self.mut_prob)

        self.population.elimination(offspring)

        return self.population.best().value, self.population.best_fitness(), self.population.mean()


class r0884600:
    # PARAMETERS
    stop = 10
    population_size = 500
    offspring_size = 1000
    k = 5
    mutation_probability = 0.8

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
        return self.no_change != self.stop

    def termination_on_fixed_iterations(self, iteration):
        return self.counter != iteration

    # The evolutionary algorithm’s main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        distanceMatrix = np.loadtxt(filename, delimiter=",")

        # Initialize the population.
        tsp = TSP(distanceMatrix, self.population_size, self.offspring_size, self.k, self.mutation_probability)

        # Run the algorithm until termination condition is met.
        while self.termination_on_best_converged():
            self.counter += 1
            self.prev_obj = self.bestObjective

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
