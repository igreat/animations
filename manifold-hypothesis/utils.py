from manim import *
from numpy import pi as PI
import math
import random
import torch
import colors
import torch.nn.functional as F
from sklearn.decomposition import PCA


def generate_outer_inner_circles(
    ax: Axes, num_dots: int = 100, dot_constructor=Dot3D
) -> tuple[tuple[Group, np.ndarray], tuple[Group, np.ndarray]]:

    # by default assumes 3d dots and 3d axes

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

    inner_dots = Group(
        *[
            dot_constructor(
                ax.c2p(*point, 0),
                radius=0.05,
                color=colors.PURPLE,
            )
            for point in inner_array
        ]
    )

    outer_dots = Group(
        *[
            dot_constructor(
                ax.c2p(*point, 0),
                radius=0.05,
                color=colors.RED,
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


# code modified from https://financial-engineering.medium.com/manim-add-brackets-584563675923
def add_brackets(mobj, shape="square"):
    shapes = {
        "square": Tex("\\big[", "\\big]"),
        "curved": Tex("\\big(", "\\big)"),
    }
    bracket_pair = shapes[shape]
    bracket_pair.stretch_to_fit_height(mobj.get_height() + 0.2)
    l_bracket, r_bracket = bracket_pair.split()
    l_bracket.next_to(mobj, LEFT, 0.005)
    r_bracket.next_to(mobj, RIGHT, 0.005)
    return VGroup(l_bracket, r_bracket)


def reduce_dimentionality(data: np.ndarray, n_components: int):
    """
    Reduces the dimentionality of a batch of data using PCA

    Parameters:
    data (np.ndarray): the batch of data to reduce the dimentionality of
    n_components (int): the number of components to reduce the data to

    Returns:
    np.ndarray: the reduced dimentionality data
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)
