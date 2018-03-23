import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def plot(x_values, y_values, y_limits = None):
    if y_limits != None:
        axes = plt.gca()
        axes.set_ylim(y_limits)
    plt.plot(x_values, y_values)
    plt.show()

def plot_points(y_values):
    x_values = list(range(len(y_values)))
    plt.plot(x_values, y_values)
    plt.show()

def plot_lattice(lattice, show=True):
    for a, b in lattice:
        plt.plot([a[0], b[0]], [a[1], b[1]], 'bo-')
    if show:
        plt.show()

def plot_heatmap(data):
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()
