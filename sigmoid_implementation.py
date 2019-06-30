# sigmoid implementation

import numpy as np
import matplotlib.pyplot as plt


def draw(x1, x2):
    plt.plot(x1, x2)


def sigmoid_function(score):
    return 1/(1+np.exp(-score))


number_points = 100
np.random.seed(0)

bias = np.ones(number_points)

above_region = np.array([np.random.normal(10, 2, number_points),
                         np.random.normal(12, 2, number_points), bias]).T
bellow_region = np.array([np.random.normal(5, 2, number_points),
                          np.random.normal(6, 2, number_points), bias]).T

# random weights
weight_1 = -0.4
weight_2 = -0.2
bias = 4.5

all_points = np.vstack((above_region, bellow_region))
line_parameters = np.matrix([weight_1, weight_2, bias]).T
x1 = np.array([bellow_region[:, 0].min(), above_region[:, 0].max()])
x2 = -bias/weight_2 + x1*(-weight_1/weight_2)
linear_combination = all_points * line_parameters


if __name__ == '__main__':

    probabilities = sigmoid_function(linear_combination)
    _, ax = plt.subplots(figsize=(6,6))

    ax.scatter(above_region[:, 0], above_region[:,1], color = 'r')
    ax.scatter(bellow_region[:, 0], bellow_region[:,1], color = 'b')

    draw(x1, x2)
    plt.show()
