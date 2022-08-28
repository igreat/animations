from manim import *
import colors
import numpy as np
from utils import *
from modules import Grid

# there are two types of animations that I can think of right now
# one is to animate the morphing
# two is to animate how the morphing changes throughout training
# I'm going to try to implement one for now

# perhaps implement adaptive scaling and shifting
# make it so that it's a 50 50 between whether the blue is over the red

# perhaps show how the last transformation both corresponds to a hyperplane or line separating the classes
# or a projection into a hyperplane or line that has a threshold that separates the classes


def change_range(range: list[ValueTracker], range_to: list[float], animation=True):
    if animation:
        return [v.animate.set_value(i) for v, i in zip(range, range_to)]
    else:
        return [v.set_value(i) for v, i in zip(range, range_to)]


def get_new_range(arrays: list[np.ndarray], num_ticks: list[int]):
    array = np.concatenate(arrays, axis=0)

    x_max, y_max = array.max(axis=0)
    x_min, y_min = array.min(axis=0)

    x_tick_len = (x_max - x_min) / num_ticks[0]
    y_tick_len = (y_max - y_min) / num_ticks[1]

    return [x_min, x_max, x_tick_len], [y_min, y_max, y_tick_len]


class SeparatingSpirals(Scene):
    def construct(self):
        self.camera.background_color = colors.WHITE

        self.x_range = [ValueTracker(-3.5), ValueTracker(3.5), ValueTracker(0.5)]
        self.y_range = [ValueTracker(-3.5), ValueTracker(3.5), ValueTracker(0.5)]

        def build_axes(x_range, y_range):
            # I want this to work also for very small values
            # for example, if the range is from 0.0001 to 0.001
            # then I want it to display it in 1e-3 to 1e-2
            # format or scientific notation

            # implement rounding to the nearest sig figs
            if type(x_range[0]) == ValueTracker:
                x_range = [round(i.get_value(), 4) for i in x_range]
            else:
                x_range = [round(i, 4) for i in x_range]

            if x_range == [0, 0, 0]:
                # in case the axis collapses into a point
                x_range = [-1, 1, 1]

            if type(y_range[0]) == ValueTracker:
                y_range = [round(i.get_value(), 4) for i in y_range]
            else:
                y_range = [round(i, 4) for i in y_range]

            if y_range == [0, 0, 0]:
                # in case the axis collapses into a point
                y_range = [-1, 1, 1]

            ax = Axes(
                x_range=x_range,
                y_range=y_range,
                x_length=7,
                y_length=7,
                tips=False,
                axis_config={
                    "unit_size": 0.5,
                    "font_size": 20,
                    "include_numbers": True,
                },
            ).set_stroke(color=colors.BLACK)

            ax.x_axis.numbers.set_stroke(colors.BLACK, 1).set_fill(colors.BLACK)
            ax.y_axis.numbers.set_stroke(colors.BLACK, 1).set_fill(colors.BLACK)

            return ax

        ax = always_redraw(lambda: build_axes(self.x_range, self.y_range))

        grid = Grid(ax, [-3.5, 3.5, 0.125], [-3.5, 3.5, 0.125])

        grid.set_color(colors.BLACK)
        grid.grid_lines.set_stroke(width=0.5)

        t = np.arange(2, 10, 0.075) * 0.4
        blue_spiral_array = (np.array([np.cos(t), np.sin(t)]) * t).T
        red_spiral_array = (np.array([np.cos(t + np.pi), np.sin(t + np.pi)]) * t).T

        blue_spiral = VGroup(
            *[
                Dot(ax.c2p(*point), radius=0.05, color=colors.PURPLE)
                for point in blue_spiral_array
            ]
        )

        red_spiral = VGroup(
            *[
                Dot(ax.c2p(*point), radius=0.05, color=colors.RED)
                for point in red_spiral_array
            ]
        )

        objects_to_transform = VGroup(grid, red_spiral, blue_spiral)

        self.play(Write(ax))
        self.play(Write(grid.grid_lines))
        self.play(Write(red_spiral))
        self.play(Write(blue_spiral))

        W1 = np.array([[-0.1118, -1.3158], [0.9703, -0.5846]])
        b1 = np.array([-0.0758, -0.0109])

        W2 = np.array([[-0.4607, -1.0789], [-0.6621, -1.1857]])
        b2 = np.array([-0.1626, -0.2111])

        W3 = np.array([[0.2765, -0.1452], [-1.2792, -1.6105]])
        b3 = np.array([0.9512, -0.4507])

        W4 = [[0.4591, -1.8305], [0.4216, -2.0197]]

        b4 = [-0.3144, -0.3060]

        # c = 8.7453
        # line = np.array([-18.8229, c]) / 10.6791

        # graph = ax.plot(lambda x: line[0] * x + line[1]).set_stroke(color=colors.BLACK)

        weights = [W1, W2, W3, W4]
        biases = [b1, b2, b3, b4]

        num_ticks = [6, 6]

        activation_funcs = [
            np.tanh,
            np.tanh,
            np.tanh,
        ]

        # something's not right...
        for w, b, func in zip(weights, biases, activation_funcs):

            # linear transformation
            blue_spiral_array = blue_spiral_array @ w.T
            red_spiral_array = red_spiral_array @ w.T
            grid.array = grid.array @ w.T

            x_range, y_range = get_new_range(
                [blue_spiral_array, red_spiral_array], num_ticks
            )
            print("x_range: ", x_range)
            print("y_range: ", y_range)

            next_ax = build_axes(x_range, y_range)

            new_blue_spiral = [
                dot.animate.move_to(next_ax.c2p(*pos))
                for dot, pos in zip(blue_spiral, blue_spiral_array)
            ]
            new_red_spiral = [
                dot.animate.move_to(next_ax.c2p(*pos))
                for dot, pos in zip(red_spiral, red_spiral_array)
            ]
            new_grid = [
                dot.animate.move_to(next_ax.c2p(*pos))
                for dot, pos in zip(grid.submobjects, grid.array)
            ]

            self.play(
                *change_range(self.x_range, x_range),
                *change_range(self.y_range, y_range),
                *new_blue_spiral,
                *new_red_spiral,
                *new_grid,
                run_time=3,
            )

            # translation
            blue_spiral_array = blue_spiral_array + b
            red_spiral_array = red_spiral_array + b
            grid.array = grid.array + b

            x_range, y_range = get_new_range(
                [blue_spiral_array, red_spiral_array], num_ticks
            )
            print("x_range: ", x_range)
            print("y_range: ", y_range)

            next_ax = build_axes(x_range, y_range)

            new_blue_spiral = [
                dot.animate.move_to(next_ax.c2p(*pos))
                for dot, pos in zip(blue_spiral, blue_spiral_array)
            ]
            new_red_spiral = [
                dot.animate.move_to(next_ax.c2p(*pos))
                for dot, pos in zip(red_spiral, red_spiral_array)
            ]
            new_grid = [
                dot.animate.move_to(next_ax.c2p(*pos))
                for dot, pos in zip(grid.submobjects, grid.array)
            ]

            self.play(
                *change_range(self.x_range, x_range),
                *change_range(self.y_range, y_range),
                *new_blue_spiral,
                *new_red_spiral,
                *new_grid,
            )

            # nonlinearity
            blue_spiral_array = func(blue_spiral_array)
            red_spiral_array = func(red_spiral_array)
            grid.array = func(grid.array)

            x_range, y_range = get_new_range(
                [blue_spiral_array, red_spiral_array], num_ticks
            )
            print("x_range: ", x_range)
            print("y_range: ", y_range)

            next_ax = build_axes(x_range, y_range)

            new_blue_spiral = [
                dot.animate.move_to(next_ax.c2p(*pos))
                for dot, pos in zip(blue_spiral, blue_spiral_array)
            ]
            new_red_spiral = [
                dot.animate.move_to(next_ax.c2p(*pos))
                for dot, pos in zip(red_spiral, red_spiral_array)
            ]
            new_grid = [
                dot.animate.move_to(next_ax.c2p(*pos))
                for dot, pos in zip(grid.submobjects, grid.array)
            ]

            self.play(
                *change_range(self.x_range, x_range),
                *change_range(self.y_range, y_range),
                *new_blue_spiral,
                *new_red_spiral,
                *new_grid,
                run_time=3,
            )

        # self.play(Write(graph))

        self.wait(2)
