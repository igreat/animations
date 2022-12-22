from manim import *
import math
import random
from utils import colors
from sklearn.decomposition import PCA


def generate_outer_inner_arrays(num_dots: int = 100) -> tuple[np.ndarray, np.ndarray]:
    inner = []
    for _ in range(num_dots):
        theta = random.random() * 2 * math.pi
        r = (random.random()) / 2
        x = math.cos(theta) * r
        y = math.sin(theta) * r
        inner.append([x, y])

    outer = []
    for _ in range(num_dots):
        theta = random.random() * 2 * math.pi
        r = (random.random()) / 2 + 0.75
        x = math.cos(theta) * r
        y = math.sin(theta) * r
        outer.append([x, y])

    inner_array = np.array(inner)
    outer_array = np.array(outer)
    return inner_array, outer_array


def generate_outer_inner_circles(
    ax: Axes, num_dots: int = 100, dot_constructor=Dot3D
) -> tuple[tuple[Group, np.ndarray], tuple[Group, np.ndarray]]:

    # by default assumes 3d dots and 3d axes

    inner_array, outer_array = generate_outer_inner_arrays(num_dots)

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
