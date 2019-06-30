#implementation just logistic regression

import numpy as np
import matplotlib.pyplot as plt

number_points = 100
np.random.seed(0)
above_region = np.array([np.random.normal(10, 2, number_points),
                         np.random.normal(12, 2, number_points)]).T
bellow_region = np.array([np.random.normal(5, 2, number_points),
                          np.random.normal(6, 2, number_points)]).T

_, ax = plt.subplots(figsize=(6,6))

ax.scatter(above_region[:, 0], above_region[:,1], color = 'r')
ax.scatter(bellow_region[:, 0], bellow_region[:,1], color = 'b')

plt.show()

