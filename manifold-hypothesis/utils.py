from manim import *
from numpy import pi as PI
import math
import random
import torch
import colors
import torch.nn.functional as F


def generate_outer_inner_circles(ax, num_dots=100):

    inner = []
    for _ in range(num_dots):
        theta = random.random() * 2 * PI
        r = (random.random()) / 2
        x = math.cos(theta) * r
        y = math.sin(theta) * r
        inner.append([x, y])

    outer = []
    for _ in range(num_dots):
        theta = random.random() * 2 * PI
        r = (random.random()) / 2 + 0.75
        x = math.cos(theta) * r
        y = math.sin(theta) * r
        outer.append([x, y])

    inner_array = np.array(inner)
    outer_array = np.array(outer)

    inner_dots = VGroup(
        *[
            Dot3D(
                ax.c2p(*point, 0),
                radius=0.05,
                color=colors.PURPLE,
                resolution=(2, 2),
            )
            for point in inner_array
        ]
    )

    outer_dots = VGroup(
        *[
            Dot3D(
                ax.c2p(*point, 0),
                radius=0.05,
                color=colors.RED,
                resolution=(2, 2),
            )
            for point in outer_array
        ]
    )
    return (inner_dots, inner_array), (outer_dots, outer_array)


def change_range(range: list[ValueTracker], range_to: list[float], animation=True):
    if animation:
        return [v.animate.set_value(i) for v, i in zip(range, range_to)]
    else:
        return [v.set_value(i) for v, i in zip(range, range_to)]


def get_new_range(arrays: list[np.ndarray], min_ticks=list[int], max_ticks=list[int]):
    array = np.concatenate(arrays, axis=0)

    x_max, y_max = array.max(axis=0)
    x_min, y_min = array.min(axis=0)

    x_diff = x_max - x_min
    y_diff = y_max - y_min

    # try to find a better mechanism for this!
    # there are clearly implications here for when x_diff is larger than 10
    num_ticks = np.array([abs((6 - x_diff)) / 6, abs((6 - y_diff)) / 6])
    num_ticks = num_ticks * max_ticks + min_ticks

    x_tick_len = x_diff / num_ticks[0]
    y_tick_len = y_diff / num_ticks[1]

    return [x_min, x_max, x_tick_len], [y_min, y_max, y_tick_len]


def leaky_relu(x):
    return F.leaky_relu(torch.tensor(x)).numpy()


def leaky_relu_inv(x, negative_slope=0.01):
    x = torch.tensor(x)
    return torch.minimum(1 / negative_slope * x, x).numpy()
