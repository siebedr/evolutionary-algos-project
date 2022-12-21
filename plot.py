import numpy as np
import matplotlib.pyplot as plt


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


def basic_plot(filename="r0884600.csv"):
    rows = parse_report(filename)

    plt.plot(rows[:, 0], rows[:, 3], label="Best")
    plt.plot(rows[:, 0], rows[:, 2], label="Mean")
    plt.ylabel('Fitness')
    plt.xlabel('Iteration')
    plt.show()


def plot_mutation_rate(filename="r0884600.csv"):
    rows = parse_report(filename)

    plt.plot(rows[:, 0], rows[:, 4], label="avg_mut_rate")
    plt.ylabel('Mutation rate')
    plt.xlabel('Iteration')
    plt.show()
