from audioop import avg, cross
from importlib.resources import path
import statistics
import Reporter
import numpy as np
import copy

# Modify the class name to match your student number.
class r0123456:
    distancematrix = np.array([])
    pop = 100
    off = 100
    k = 3
    stop = 15
    no_change = 0
    counter = 0

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):

        # Read distance matrix from file.		
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # print(self.distanceMatrix)

        population = self.initialize_pop(len(self.distanceMatrix))

        # for i in population:
        #     print(self.path_cost(i))

        prev_obj = 0
        bestObjective = 0.0

        # Your code here.
        while( self.termination() ):
            self.counter += 1
            print("iteration: " + str(self.counter))
            prev_obj = bestObjective
            meanObjective = 0.0
            bestObjective = 0.0
            bestSolution = np.array([1,2,3,4,5])
            costs = []

            offspring = []

            for i in range(self.pop):
                offspring.append(self.mutation(copy.deepcopy(population[i])))
    
            
            population, offspring = self.crossover(population, offspring)

            population = self.elimination(population + offspring)


            for i in population:
                costs.append(self.path_cost(i))

            meanObjective = statistics.mean(costs)
            bestObjective = min(costs)
            bestSolution = population[np.argmin(costs)]

            if (prev_obj == bestObjective):
                self.no_change += 1
            else:
                self.no_change = 0

            print(meanObjective)
            print(bestObjective)
            print(bestSolution)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution 
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        return 0
    
    def termination(self):        
        return self.no_change != self.stop

    def initialize_pop(self, pathlength):
        population = []

        while len(population) < self.pop:
            randpath = np.random.permutation(pathlength)
            if str(self.path_cost(randpath)) != "inf":
                population.append(randpath)

        print(len(population))
        return population
    
    def path_cost(self, path):
        total = 0
        for i in range(len(path)):
            if (i == len(path)-1):
                cost = self.distanceMatrix[path[i]][0]
            else:
                cost = self.distanceMatrix[path[i]][path[i+1]]

            if cost == "inf":
                return "inf"
            else:
                total += cost
        return total

    def mutation(self, path):
        prob = 0.05

        for i in range(len(path)):
            if (np.random.random() < prob):
                swap = path[i]
                new_index = np.random.randint(len(path))
                path[i] = path[new_index]
                path[new_index] = swap
        
        return path

    def selection(self, population):
        selected = []

        for ii in range( self.off ):
            ri = np.random.choices(range(np.size(population,0)), k = self.k)
            min = np.argmin( self.objf(population[ri, :]) )
            selected[ii,:] = population[ri[min],:]
        return selected

    def crossover(self, population, offspring):
        prob = 0.6

        for i in range(len(population)):
            if (np.random.random() < prob):
                crossover_pos = np.random.randint(0, self.pop-1)
                p1 = population[i]
                p2 = offspring[i]
                c1 = p1[:crossover_pos]
                c2 = p2[:crossover_pos]

                for j1 in p1:
                    if j1 not in c2:
                        c2 = np.append(c2, j1)

                for j2 in p2:
                    if j2 not in c1:
                        c1 = np.append(c1, j2)

                population[i] = c1
                offspring[i] = c2        

        return population, offspring

    def elimination(self, population):
        best = []
        worst = []

        for i in population:
            if str(self.path_cost(i)) == "inf":
                worst.append(i)
            else:
                best.append(i)

        best.sort(key=self.path_cost)
        new_pop = []

        if len(best) <= (self.pop//2):
            new_pop = best
        else:
            new_pop = best[:(self.pop//2)]
            worst = worst + best[(self.pop//2):]

        while len(new_pop) < self.pop:
            index = np.random.randint(len(worst))
            rand_inst = worst[index]
            del worst[index]
            new_pop.append(rand_inst)

        return new_pop
            

program = r0123456()
program.optimize("./Data/tour50.csv")