import random
import time

import numpy as np
from numba import njit

import Reporter


# NUMBA functions to improve speed
@njit
def swap_mutation(path, mutation_rate):
    if np.random.rand() < mutation_rate:
        new_index1 = np.random.randint(len(path))
        new_index2 = np.random.randint(len(path))

        swap = path[new_index1]
        path[new_index1] = path[new_index2]
        path[new_index2] = swap

    return path


@njit
def inverse_mutation(path, mutation_rate):
    if np.random.rand() < mutation_rate:
        rand1 = np.random.randint(len(path))
        rand2 = np.random.randint(len(path))

        start_node = min(rand1, rand2)
        end_node = max(rand1, rand2)

        path[start_node:end_node] = path[start_node:end_node][::-1]

    return path


@njit
def recombine_self_adaptive_rates(rate1, rate2):
    # Calculate combined mutation rate
    beta = 2 * np.random.rand() - 0.5
    return max(0.1, rate1 + beta * ((rate2 - rate1) - np.random.rand() * 0.005))


@njit
def ordered_crossover(path1, path2):
    rand1 = np.random.randint(len(path1))
    rand2 = np.random.randint(len(path1))

    start_node = min(rand1, rand2)
    end_node = max(rand1, rand2)

    other_values = [x for x in path2 if x not in path1[start_node:end_node]]

    child = np.empty(len(path1), dtype=np.int32)

    for i in range(start_node):
        child[i] = other_values.pop(0)

    child[start_node:end_node] = path1[start_node:end_node]

    for i in range(end_node, len(path1)):
        child[i] = other_values.pop(0)

    return child


@njit
def PMX_crossover(path1, path2):
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

    return child


@njit
def cycle_crossover(path1, path2):
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

    return child


@njit
def light_two_opt(path, dist, depth=-1):
    # Pre calculate distances
    dists = np.zeros(len(path))
    reverse_dists = np.zeros(len(path))

    for i in range(len(path)):
        dists[i] = dist[path[i], path[(i + 1) % len(path)]]
        reverse_dists[i] = dist[path[(i + 1) % len(path)], path[i]]

    best_dist = cost_helper(path, dist)
    best = path
    counter = 0

    for i in range(1, len(path)):
        for j in range(i + 2, len(path)):
            fast_cost = np.sum(dists[:i - 1]) + np.sum(dists[j + 1:]) + np.sum(reverse_dists[i:j]) \
                        + dist[path[i - 1], path[j]] + dist[path[i], path[(j + 1) % len(path)]]

            if fast_cost <= best_dist:
                best = np.concatenate((path[:i], path[i:j + 1][::-1], path[j + 1:]))
                best_dist = fast_cost
                counter += 1

                if depth != -1 and counter >= depth:
                    return best
    return best


@njit
def heuristic_ls(route, dist_matrix, max_size=None):
    if max_size is None:
        rand1 = np.random.randint(1, len(route))
        rand2 = np.random.randint(1, len(route))

        start_node = min(rand1, rand2)
        end_node = max(rand1, rand2)
    else:
        rand1 = np.random.randint(1, len(route) - max_size)
        start_node = rand1
        end_node = np.random.randint(start_node, start_node + max_size)

    best = route
    best_cost = sum([dist_matrix[route[x]][route[x + 1]] for x in range(start_node - 1, end_node)])

    for i in range(start_node, end_node):
        possibilities = set(route[start_node:end_node])
        sub_path = np.empty(len(possibilities), dtype=np.int32)
        cost = 0

        for j in range(end_node - start_node):
            best_node = -1
            best_cost = np.inf
            for x in possibilities:
                if dist_matrix[route[j - 1]][x] < best_cost:
                    best_node = x
                    best_cost = dist_matrix[route[j - 1]][x]

            if best_node != -1:
                sub_path[j] = best_node
                possibilities.remove(best_node)
                cost += best_cost
            else:
                break

        if len(possibilities) == 0 and cost < best_cost:
            return np.concatenate((route[:start_node], sub_path, route[end_node:]))

    return best


@njit
def greedy_insert_ls(route, dist_matrix):
    city_pos = np.random.randint(1, len(route) - 1)
    cost = dist_matrix[route[city_pos - 1]][route[city_pos]] + dist_matrix[route[city_pos]][route[city_pos + 1]]
    new_pos = city_pos

    route_without = np.concatenate((route[:city_pos], route[city_pos + 1:]))

    for i in range(1, len(route_without)):
        c = dist_matrix[route[i - 1]][route[city_pos]] + dist_matrix[route[city_pos]][route[i]]
        if c < cost:
            new_pos = i

    new_route = route_without[:new_pos]
    new_route = np.append(new_route, route[city_pos])

    return np.concatenate((new_route, route_without[new_pos:]))


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
    counter = 0
    iter1 = np.where(route1 == 0)[0]
    iter2 = np.where(route2 == 0)[0]

    for i in range(len(route1)):
        if route1[iter1] == route2[iter2]:
            counter += 1
        iter1 = (iter1 + 1) % len(route1)
        iter2 = (iter2 + 1) % len(route2)

    return (len(route1) - counter) / len(route1)


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
def shared_elimination(population, dist_matrix, keep, elites, sigma=0.1):
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
                fitness[j] = fitness_sharing_with_list(surviving_route_distances[:i, j], fitness[j], sigma)

            relevant.clear()

        survivors[i] = np.argmax(fitness)
        battling.remove(survivors[i])
        fitness[survivors[i]] = -1
        best_route_distances = calc_route_distances(battling, population, population[survivors[i]])
        surviving_route_distances[i] = best_route_distances
        [relevant.add(y) for y, x in enumerate(best_route_distances) if x <= sigma]

    return survivors


@njit
def init_NN_path(dist_matrix):
    """Initial population diversification by adding nearest neighbour paths, k can be used to not be too greedy"""
    found = False
    path = np.empty(len(dist_matrix), dtype=np.int32)

    if 250 < len(dist_matrix) < 700:
        k = len(dist_matrix) // 10
    elif len(dist_matrix) >= 700:
        k = len(dist_matrix) // 4
    else:
        k = len(dist_matrix) // 50

    while not found:
        possible_nodes = set([i for i in range(len(dist_matrix))])
        path[0] = np.random.randint(len(dist_matrix))
        possible_nodes.remove(path[0])
        found = True

        for i in range(1, len(dist_matrix)):
            best = np.inf
            best_node = -1

            perm = np.random.permutation(np.array(list(possible_nodes)))
            k_nodes = perm[:k]

            for j in k_nodes:
                if dist_matrix[path[i - 1]][j] < best:
                    best = dist_matrix[path[i - 1]][j]
                    best_node = j

            if best_node == -1:
                found = False
                break

            path[i] = best_node
            possible_nodes.remove(best_node)

    return path


def init_random_legal_path(dist_matrix):
    """Initial population diversification by legal paths"""
    found = False
    path = np.empty(len(dist_matrix), dtype=np.int32)

    while not found:
        possible_nodes = set([i for i in range(len(dist_matrix))])
        path[0] = np.random.randint(len(dist_matrix))
        possible_nodes.remove(path[0])
        found = True

        for i in range(1, len(dist_matrix)):
            node = -1
            choices = np.array(list(possible_nodes))
            for j in possible_nodes:
                possible = np.random.choice(choices, replace=False)
                if dist_matrix[path[i - 1]][possible] != float('inf'):
                    node = possible
                    break

            if node == -1:
                found = False
                break

            path[i] = node
            possible_nodes.remove(node)

            if len(possible_nodes) == 0 and dist_matrix[path[-1]][path[0]] == float('inf'):
                found = False
                break

    return path


class Individual:
    def __init__(self, value, mutation_rate=None, exploit_rate=None):
        self.value = value
        if mutation_rate is None:
            self.mutation_rate = 0.1 + (0.3 * np.random.rand())
        else:
            self.mutation_rate = mutation_rate
        self.neighbour_dist = None

        if exploit_rate is None:
            self.exploit_rate = 0.3 + (0.3 * np.random.rand())
        else:
            self.exploit_rate = exploit_rate

    def mutate(self):
        self.value = inverse_mutation(self.value, self.mutation_rate)

    def local_search_operator(self, dist):
        rand = np.random.rand()

        if rand < self.exploit_rate:
            if len(dist) > 500:
                self.value = greedy_insert_ls(self.value, dist)
            else:
                self.value = light_two_opt(self.value, dist)
        else:
            if len(dist) > 500:
                self.value = heuristic_ls(self.value, dist, len(dist) // 4)
            else:
                self.value = heuristic_ls(self.value, dist, len(dist) // 2)

    def recombine(self, other):
        # Ordered crossover, returns child
        child = ordered_crossover(self.value, other.value)
        mut_rate = recombine_self_adaptive_rates(self.mutation_rate, other.mutation_rate)
        exp_rate = recombine_self_adaptive_rates(self.exploit_rate, other.exploit_rate)
        return Individual(child, mut_rate, exp_rate)


class Population:
    def __init__(self, size, dist_matrix, elites, random_init):
        self.route_dist_matrix = None
        self.size = size
        self.dist_matrix = dist_matrix

        self.individuals: list[Individual] = []
        self.init_population(random_init)
        self.elites = int(size * elites)

    def init_population(self, random_part=0.7, heuristic=0.5):
        # Random initialization
        random_size = int(self.size * random_part)
        heuristic_part = int((self.size - random_size) * heuristic)
        others = self.size - random_size - heuristic_part

        while len(self.individuals) < random_size:
            self.individuals.append(Individual(np.random.permutation(len(self.dist_matrix))))

        heuristics = []
        while len(heuristics) < heuristic_part * 4:
            heuristics.append(init_NN_path(self.dist_matrix))

        survivors = shared_elimination(np.array(heuristics), self.dist_matrix, heuristic_part, 0, 0.6)
        [self.individuals.append(Individual(heuristics[x])) for x in survivors]

        while len(self.individuals) < random_size + others:
            self.individuals.append(Individual(init_random_legal_path(self.dist_matrix)))

    def elimination(self, offspring: list[Individual]):
        # Does elimination and replaces original population (alpha + mu)
        offspring += self.individuals[self.elites:]
        offspring.sort(key=lambda x: self.fitness(x), reverse=True)
        self.individuals = offspring[:(self.size - self.elites)] + self.individuals[:self.elites]
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

    def __init__(self, distance_matrix, population_size, offspring_size, k, elites, random_init):
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.k = k

        self.population = Population(self.population_size, self.distance_matrix, elites, random_init)

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

        while len(offspring) < self.offspring_size:
            mother: Individual = self.population.selection(self.k)
            father: Individual = self.population.selection(self.k)
            child = mother.recombine(father)
            child.mutate()
            offspring.append(child)

        self.population.mutate_all()

        self.ls_all(offspring)

        self.population.fs_elimination(offspring)
        return self.population.best(), self.population.best_fitness(), self.population.mean()


class r0884600:
    # PARAMETERS
    stop = 50
    population_size = 30
    offspring_size = 50
    k = 5
    elites = 0.1

    random_init = 0.7

    no_change = 0
    counter = 0

    meanObjective = 0.0
    bestObjective = 0.0
    bestSolution: Individual = None
    prev_obj = 0

    log_interval = 10

    def __init__(self, name=None):
        if name is None:
            self.reporter = Reporter.Reporter(self.__class__.__name__)
        else:
            self.reporter = Reporter.Reporter(name)

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

    def termination_on_mean_diff_converged(self):
        if self.bestObjective - self.meanObjective < self.prev_obj:
            self.no_change += 1
        else:
            self.no_change = 0

        self.prev_obj = self.bestObjective - self.meanObjective

        return self.no_change != self.stop

    def termination_on_fixed_iterations(self, iteration):
        return self.counter != iteration

    # The evolutionary algorithm’s main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        distanceMatrix = np.loadtxt(filename, delimiter=",")

        if len(distanceMatrix) >= 100:
            self.stop = len(distanceMatrix) // 2

        if len(distanceMatrix) > 400:
            self.population_size = 10
            self.offspring_size = 20
        else:
            self.population_size = 25
            self.offspring_size = 50

        # Initialize the population.
        tsp = TSP(distanceMatrix, self.population_size, self.offspring_size, self.k, self.elites, self.random_init)

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
