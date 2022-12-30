import numpy as np
import matplotlib.pyplot as plt
import statistics


# Parse report file
# Iteration, Elapsed time, Mean value, Best value, Cycle
def parse_report(filename):
    file_rows = open(filename, 'r').readlines()
    rows = []
    for i in range(2, len(file_rows)):
        tmp_row = file_rows[i].split(',')[:-1]
        typed_row = []

        for j in range(len(tmp_row)):
            if j == 0 or j >= 5:
                typed_row.append(int(tmp_row[j]))
            else:
                typed_row.append(float(tmp_row[j]))

        rows.append(typed_row)

    return np.array(rows)


def basic_plot(title, filename="r0884600.csv"):
    rows = parse_report(filename)

    plt.plot(rows[:, 0], rows[:, 3], label="Best")
    plt.plot(rows[:, 0], rows[:, 2], label="Mean")
    plt.ylabel('Fitness')
    plt.xlabel('Iteration')
    plt.title(title)
    plt.show()


def plot_mutation_rate(filename="100.csv"):
    rows = parse_report(filename)

    plt.plot(rows[:, 0], rows[:, 4], label="avg_mut_rate")
    plt.ylabel('Mutation rate')
    plt.xlabel('Iteration')
    plt.show()


def plot_hist(title, iterations=1000):
    scores = []

    for i in range(iterations):
        rows = parse_report("./Tests/test_" + str(i) + ".csv")
        scores.append(1/rows[-1, 2])

    plt.hist(scores, 'auto', alpha=0.7, rwidth=0.85)
    plt.axvline(statistics.mean(scores), color='r', linestyle='dashed', linewidth=1.2)

    plt.ylabel('Frequency')
    plt.xlabel('Best score')
    plt.title(title)
    plt.show()

    print(statistics.stdev(scores))
    print(statistics.mean(scores))
    print(scores.index(min(scores)))


# plot_hist("Histogram of mean scores of 1000 runs on tour 50", 1000)
#basic_plot("Best run on tour 50", "./Tests/test_759.csv")