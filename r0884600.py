import Reporter
import numpy as np
import random

class Individual():
    def __init__(self, value):
        self.value = value

    def mutate(self, prob):
        # Swap mutation
        if (np.random.random() < prob):
            new_index1 = np.random.randint(len(self.value))
            new_index2 = np.random.randint(len(self.value))

            swap = self.value[new_index1]
            self.value[new_index1] = self.value[new_index2]
            self.value[new_index2] = swap
        
    def recombine(self, other):
        # Ordered corssrover, returns child
        rand1 = np.random.randint(len(self.value))
        rand2 = np.random.randint(len(self.value))

        start = min(rand1, rand2)
        end = max(rand1, rand2)

        other_values = [x for x in other.value if x not in self.value[start:end]]

        child = []

        for i in range(start):
            child.append(other_values.pop(0))

        for i in range(start, end):
            child.append(self.value[i])

        for i in range(end, len(self.value)):
            child.append(other_values.pop(0))

        return Individual(np.array(child))

class Population():
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
        possibile = random.choices(self.individuals, k=k)
        possibile.sort(key=lambda x: self.fitness(x), reverse=True)
        return possibile[0]

    def mutate_all(self, prob):
        for ind in self.individuals:
            ind.mutate(prob)

    def fitness(self, individual : Individual):
        return 1/self.cost(individual)

    def cost(self, individual : Individual):
        total = 0

        for i in range(len(individual.value)-1):
            val = individual.value[i]
            next = individual.value[i+1]
            total += self.dist_matrix[val][next]

        return total + self.dist_matrix[individual.value[len(individual.value)-1]][0]

    def best(self) -> Individual:
        return self.individuals[0]

    def best_fitness(self) -> float:
        return self.fitness(self.individuals[0])
    
    def mean(self)-> Individual:
        total_fitness = 0
        for ind in self.individuals:
            total_fitness += self.fitness(ind)

        return total_fitness / self.size
  

class TSP():

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
    population_size=500
    offspring_size=1000
    k=5
    mutation_probability=0.8

    no_change = 0
    counter = 0

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def termination(self):        
        return self.no_change != self.stop

    # The evolutionary algorithm’s main loop
    def optimize(self, filename):
        meanObjective = 0.0 
        bestObjective = 0.0
        bestSolution : Individual = None

        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",") 
        file.close()

        tsp = TSP(distanceMatrix, self.population_size, self.offspring_size, self.k, self.mutation_probability)

        while( self.termination() ):
            self.counter += 1
            prev_obj = bestObjective

            bestSolution, bestObjective, meanObjective = tsp.step()
            
            if (prev_obj == bestObjective):
                self.no_change += 1
            else:
                self.no_change = 0
            
            timeLeft = self.reporter.report(meanObjective , bestObjective , bestSolution)
            if timeLeft < 0: 
                break
            
            print("\nIteration: ", self.counter, "\nMean: ", meanObjective, "\nBest: ", bestObjective, "\nPath cost: ", 1/bestObjective)
        return 0

program = r0884600()
program.optimize("./Data/tour250.csv")