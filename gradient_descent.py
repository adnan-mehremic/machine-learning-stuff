# gradient descent / linear regression

import numpy as np
import matplotlib.pyplot as plt


def draw(x1, x2):
    plt.plot(x1, x2)


def sigmoid(score):
    return 1/(1 + np.exp(-score))


def calculate_error(line_parameters, points, y):
    n = points.shape[0]
    p = sigmoid(points * line_parameters)
    cross_entropy = -(1 / n) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return cross_entropy


def gradient_descent(line_parameters, points, y, alpha):
    n = points.shape[0]
    for i in range(1500):
        p = sigmoid(points * line_parameters)
        gradient = points.T * (p - y) * (alpha / n)
        line_parameters = line_parameters - gradient

        weight_1 = line_parameters.item(0)
        weight_2 = line_parameters.item(1)
        b = line_parameters.item(2)

        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / weight_2 + (x1 * (-weight_1 / weight_2))

    draw(x1, x2)


if __name__ == '__main__':

    number_points = 100
    np.random.seed(0)
    bias = np.ones(number_points)
    above_region = np.array([np.random.normal(10, 2, number_points),
                           np.random.normal(12, 2, number_points), bias]).T
    bellow_region = np.array([np.random.normal(5, 2, number_points),
                              np.random.normal(6, 2, number_points), bias]).T
    all_points = np.vstack((above_region, bellow_region))

    line_parameters = np.matrix([np.zeros(3)]).T
    y = np.array([np.zeros(number_points), np.ones(number_points)]).reshape(number_points * 2, 1)

    _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(above_region[:, 0], above_region[:, 1], color='r')
    ax.scatter(bellow_region[:, 0], bellow_region[:, 1], color='b')
    gradient_descent(line_parameters, all_points, y, 0.05)
    plt.show()