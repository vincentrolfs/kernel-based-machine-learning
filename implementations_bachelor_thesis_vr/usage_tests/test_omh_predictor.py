import numpy as np
import matplotlib.pyplot as plt

from implementations_bachelor_thesis_vr.omh_predictor import Omh_Predictor

def plot_inputs(ax, inputs, style):
    inputs_x = [input[0] for input in inputs]
    inputs_y = [input[1] for input in inputs]
    ax.plot(inputs_x, inputs_y, style)

red_inputs = np.array([
    [-3, 1.5],
    [-1, 2],
    [1, 2],
    [2.5, 2.5],
    [2.5, 4]
])
green_inputs = np.array([
    [-1, -1],
    [1, 0],
    [2, 1],
    [3.5, 0],
    [5, 1]
])
all_inputs = np.concatenate( (red_inputs, green_inputs) )
labels = np.array([1 for _ in red_inputs] + [-1 for _ in green_inputs])

predictor = Omh_Predictor(all_inputs, labels)
predictor.train(tolerance = 10**(-5))
v = predictor.v
s = predictor.s

print("v = ", v)
print("s = ", s)

fig, ax = plt.subplots()

plot_inputs(ax, red_inputs, 'ro')
plot_inputs(ax, green_inputs, 'go')

hyperplane_x = np.arange(-5, 7)
hyperplane_y = [(x*v[0] + s)/(-v[1]) for x in hyperplane_x]
ax.plot(hyperplane_x, hyperplane_y)

plt.axis('equal')
plt.xticks(np.arange(-5, 7))
plt.yticks(np.arange(-5, 7))
plt.grid()

ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.show()
