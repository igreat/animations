from re import L
from manim import *
import colors
import numpy as np
from utils import *
from modules import *
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

config.background_color = colors.WHITE

# try to perhaps animate how points classified by the network morph and shift and how they make a clean boundary in the final layer!
# this is the best workaround I can think of to the failure to make reverse transformations work...

# perhaps show how the last transformation both corresponds to a hyperplane or line separating the classes


def generate_outer_inner_circles(ax, num_dots=100):

    from numpy import pi as PI
    import math

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


class SeparatingSpirals2d(Scene):
    def construct(self):
        self.camera.background_color = colors.WHITE

        self.build_spirals()
        self.wait()

        self.model = Spirals2dModel().eval()
        self.model.load_state_dict(torch.load("saved_models/separating_spirals2d_tanh"))

        # think about displaying the matrices as the transformation goes on
        self.transform()
        self.wait(2)
        # self.reverse_transform()

    def build_spirals(self):
        self.x_range = [ValueTracker(-3.5), ValueTracker(3.5), ValueTracker(0.5)]
        self.y_range = [ValueTracker(-3.5), ValueTracker(3.5), ValueTracker(0.5)]

        def build_axes(x_range, y_range):

            if type(x_range[0]) == ValueTracker:
                x_range = [i.get_value() for i in x_range]

            if x_range == [0, 0, 0]:
                # in case the axis collapses into a point
                x_range = [0, 1, 0.25]

            if type(y_range[0]) == ValueTracker:
                y_range = [i.get_value() for i in y_range]

            if y_range == [0, 0, 0]:
                # in case the axis collapses into a point
                y_range = [0, 1, 0.25]

            ax = Axes(
                x_range=x_range,
                y_range=y_range,
                x_length=7,
                y_length=7,
                tips=False,
                axis_config={
                    "unit_size": 0.5,
                    "font_size": 20,
                },
            ).set_stroke(color=colors.BLACK)

            return ax

        self.ax = build_axes(self.x_range, self.y_range)
        # ax = always_redraw(lambda: build_axes(self.x_range, self.y_range))

        # grid = Grid(ax, [-3.5, 3.5, 0.5], [-3.5, 3.5, 0.5])

        # grid.set_color(colors.BLACK)
        # grid.grid_lines.set_stroke(width=0.5)

        t = np.arange(2, 10, 0.1) * 0.4
        self.blue_spiral_array = (np.array([np.cos(t), np.sin(t)]) * t).T
        self.red_spiral_array = (np.array([np.cos(t + np.pi), np.sin(t + np.pi)]) * t).T

        self.blue_spiral = VGroup(
            *[
                Dot(self.ax.c2p(*point), radius=0.05, color=colors.PURPLE)
                for point in self.blue_spiral_array
            ]
        )

        self.red_spiral = VGroup(
            *[
                Dot(self.ax.c2p(*point), radius=0.05, color=colors.RED)
                for point in self.red_spiral_array
            ]
        )

        self.play(Write(self.ax))
        # self.play(Write(grid.grid_lines))
        self.play(Write(self.red_spiral))
        self.play(Write(self.blue_spiral))

    def transform(self):

        blue_outputs = self.model(torch.tensor(self.blue_spiral_array).float())
        red_outputs = self.model(torch.tensor(self.red_spiral_array).float())
        # grid.array

        for blue_output, red_output in zip(blue_outputs[:-1], red_outputs[:-1]):
            new_blue_spiral = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(self.blue_spiral, blue_output.detach())
            ]
            new_red_spiral = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(self.red_spiral, red_output.detach())
            ]
            # new_grid = [
            #     dot.animate.move_to(next_ax.c2p(*pos))
            #     for dot, pos in zip(grid.submobjects, grid.array)
            # ]

            self.play(
                *new_blue_spiral,
                *new_red_spiral,
                # *new_grid,
                run_time=3,
                rate_func=rate_functions.linear,
            )

        self.wait()

        wf, bf = list(self.model.parameters())[-2:]
        wf, bf = wf.squeeze().detach(), bf.squeeze().detach()

        # perhaps make it clear that that I'm scaling it down to be visible in screen
        scale_factor = 2e-1
        new_blue_spiral = [
            dot.animate.move_to(self.ax.c2p(pos * scale_factor, 0))
            for dot, pos in zip(self.blue_spiral, blue_outputs[-1].detach())
        ]
        new_red_spiral = [
            dot.animate.move_to(self.ax.c2p(pos * scale_factor, 0))
            for dot, pos in zip(self.red_spiral, red_outputs[-1].detach())
        ]

        self.play(
            *new_blue_spiral,
            *new_red_spiral,
            # *new_grid,
            rate_func=rate_functions.linear,
        )
        self.wait()

        new_blue_spiral = [
            dot.animate.move_to(self.ax.c2p(*pos))
            for dot, pos in zip(self.blue_spiral, blue_outputs[-2].detach())
        ]
        new_red_spiral = [
            dot.animate.move_to(self.ax.c2p(*pos))
            for dot, pos in zip(self.red_spiral, red_outputs[-2].detach())
        ]

        self.play(
            *new_blue_spiral,
            *new_red_spiral,
            # *new_grid,
            rate_func=rate_functions.linear,
        )

        # go a little deeper into explaining this part
        line = Line(
            *[self.ax.c2p(x, -wf[0] / wf[1] * x - bf / wf[1]) for x in [-2.5, 2.5]]
        ).set_stroke(color=colors.BLACK)

        self.play(Create(line))

    # this part seems to not work at all...
    def reverse_transform(self):

        reverse_weights = [np.linalg.inv(w) for w in reversed(self.weights[:-1])]
        reverse_biases = [-b for b in reversed(self.biases[:-1])]
        reverse_acts = [np.arctanh for _ in range(4)]

        # here also put the boundary line!
        # THERE IS SOMETHING FUNDEMENTALLY WRONG WITH THE SHIFT/BIAS THING IN THE BOUNDARY LINE, FIX IT SOON
        W5, b5 = self.weights[-1].squeeze(), self.biases[-1].squeeze()
        line = DotsPlot(self.ax, lambda x: -W5[0] / W5[1] * x - b5 / W5[1], -1, 1, 1000)
        self.play(Create(line.sublines))

        for w, b, func in zip(reverse_weights, reverse_biases, reverse_acts):

            # nonlinearity
            self.blue_spiral_array = func(self.blue_spiral_array)
            self.red_spiral_array = func(self.red_spiral_array)
            # grid.array = func(grid.array)
            line.array = func(line.array)

            new_blue_spiral = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(self.blue_spiral, self.blue_spiral_array)
            ]
            new_red_spiral = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(self.red_spiral, self.red_spiral_array)
            ]
            # new_grid = [
            #     dot.animate.move_to(next_ax.c2p(*pos))
            #     for dot, pos in zip(grid.submobjects, grid.array)
            # ]

            new_line = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(line.submobjects, line.array)
            ]

            self.play(
                *new_blue_spiral,
                *new_red_spiral,
                # *new_grid,
                *new_line,
                run_time=3,
                rate_func=rate_functions.linear,
            )

            # translation
            self.blue_spiral_array += b
            self.red_spiral_array += b
            # grid.array = grid.array + b
            line.array = line.array + b

            new_blue_spiral = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(self.blue_spiral, self.blue_spiral_array)
            ]
            new_red_spiral = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(self.red_spiral, self.red_spiral_array)
            ]
            # new_grid = [
            #     dot.animate.move_to(next_ax.c2p(*pos))
            #     for dot, pos in zip(grid.submobjects, grid.array)
            # ]

            new_line = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(line.submobjects, line.array)
            ]

            self.play(
                *new_blue_spiral,
                *new_red_spiral,
                # *new_grid,
                *new_line,
                rate_func=rate_functions.linear,
            )

            # linear transformation
            self.blue_spiral_array = self.blue_spiral_array @ w.T
            self.red_spiral_array = self.red_spiral_array @ w.T
            # grid.array = grid.array @ w.T
            line.array = line.array @ w.T

            new_blue_spiral = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(self.blue_spiral, self.blue_spiral_array)
            ]
            new_red_spiral = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(self.red_spiral, self.red_spiral_array)
            ]
            # new_grid = [
            #     dot.animate.move_to(self.ax.c2p(*pos))
            #     for dot, pos in zip(grid.submobjects, grid.array)
            # ]

            new_line = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(line.submobjects, line.array)
            ]

            self.play(
                *new_blue_spiral,
                *new_red_spiral,
                # *new_grid,
                *new_line,
                run_time=3,
                rate_func=rate_functions.linear,
            )

        self.wait(2)


class SeparatingSpirals2dTraining(Scene):
    def construct(self):
        self.camera.background_color = colors.WHITE
        x_range = [-2.5, 2.5, 0.5]
        y_range = [-2.5, 2.5, 0.5]

        # here what I actually want is multiple axis arranged in a grid (use .arrange(rows, cols))
        # each grid will be for a particular layer

        ax_grid = VGroup()
        for i in range(8):
            ax = Axes(
                x_range=x_range,
                y_range=y_range,
                x_length=10 / 3,
                y_length=10 / 3,
                tips=False,
                axis_config={"tick_size": 0.05},
            ).set_stroke(color=colors.BLACK)
            ax_grid.add(ax)
        ax_grid.arrange_in_grid(2, buff=0.05)

        t = np.arange(1, 5, 0.05)
        array1 = (np.array([np.cos(t), np.sin(t)]) * t).T * 0.5
        array2 = (np.array([np.cos(t + np.pi), np.sin(t + np.pi)]) * t).T * 0.5

        self.layers = []

        dots_all = VGroup()
        for ax in ax_grid:
            dots1 = VGroup(
                *[
                    Dot(ax.c2p(*point), radius=0.04, color=colors.PURPLE)
                    for point in array1
                ]
            )

            dots2 = VGroup(
                *[
                    Dot([ax.c2p(*point)], radius=0.04, color=colors.RED)
                    for point in array2
                ]
            )
            dots_all.add(VGroup(dots1, dots2))

        self.play(Write(VGroup(ax_grid)))

        # setting up data for training
        labels1 = np.ones(len(array1)).reshape(-1, 1)
        labels2 = np.zeros(len(array2)).reshape(-1, 1)

        class1_data = np.concatenate(
            (array1.T, labels1.T, np.arange(len(array1)).reshape(1, -1)), axis=0
        )
        class2_data = np.concatenate(
            (array2.T, labels2.T, np.arange(len(array2)).reshape(1, -1)), axis=0
        )

        # consider adding a shuffling mechanism
        data = np.concatenate((class1_data, class2_data), axis=1)
        data = torch.tensor(data, dtype=torch.float32).T

        # setting up the model
        model = TrainingModel().float()

        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        model.requires_grad_(True)

        # getting the initial position
        x, labels, indices = (
            data[:, 0:2],
            data[:, 2].unsqueeze(1),
            data[:, 3].unsqueeze(1),
        )
        pred, hidden_out = model(x)
        with torch.no_grad():
            for ax, dots, layer_num in zip(ax_grid, dots_all, range(8)):
                layer = hidden_out[layer_num].numpy()
                for h_out, label, i in zip(layer, labels, indices):
                    if label == 0:
                        dots[0][int(i.item())].move_to(ax.c2p(*h_out))
                    else:
                        dots[1][int(i.item())].move_to(ax.c2p(*h_out))

        self.play(Write(dots_all))

        # training loop
        for epoch in range(200):
            x, labels, indices = (
                data[:, 0:2],
                data[:, 2].unsqueeze(1),
                data[:, 3].unsqueeze(1),
            )
            pred, hidden_out = model(x)
            loss = F.binary_cross_entropy_with_logits(pred, labels)
            with torch.no_grad():
                animations = []
                for ax, dots, layer_num in zip(ax_grid, dots_all, range(8)):
                    layer = hidden_out[layer_num].numpy()
                    for h_out, label, i in zip(layer, labels, indices):
                        if label == 0:
                            animations.append(
                                dots[0][int(i.item())].animate.move_to(ax.c2p(*h_out))
                            )
                        else:
                            animations.append(
                                dots[1][int(i.item())].animate.move_to(ax.c2p(*h_out))
                            )

                if epoch % 1 == 0:
                    print(f"epoch: {epoch}, loss: {loss.item():.4f}")
            self.play(*animations, run_time=0.005, rate_func=linear)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.wait(2)


class SeparatingSpirals3d(ThreeDScene):
    def construct(self):
        self.camera.background_color = colors.WHITE
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        x_range = [-2.5, 2.5, 0.5]
        y_range = [-2.5, 2.5, 0.5]

        ax = ThreeDAxes(
            x_range=x_range,
            y_range=y_range,
            x_length=7,
            y_length=7,
            tips=False,
        ).set_stroke(color=colors.BLACK)

        (inner_dots, inner_array), (
            outer_dots,
            outer_array,
        ) = generate_outer_inner_circles(ax, 100)

        self.begin_ambient_camera_rotation(-0.1)

        self.play(Write(ax))
        self.play(Write(inner_dots))
        self.play(Write(outer_dots))

        # linear transformation

        W1 = np.array(
            [
                [-1.3335, -1.3497],
                [-0.2760, 1.2892],
                [1.6805, -0.7282],
            ],
        )
        b1 = np.array([-0.8905, -0.5729, -0.8609])

        W2 = np.array(
            [
                [1.3766, 0.8067, 1.3736],
                [0.9967, 1.8312, 1.1230],
                [0.9773, 1.3226, 0.9705],
            ],
        )
        b2 = np.array([1.4654, 1.6515, 1.4718])

        W3 = np.array(
            [
                [-1.1099, -1.3832, -1.7226],
                [-1.3346, -1.3959, -0.7877],
                [1.1106, 1.2383, 2.1721],
            ],
        )
        b3 = np.array([-0.5693, -0.5870, 0.5870])

        W4 = np.array(
            [
                [1.3508, 1.4681, -1.7920],
                [1.5001, 1.6317, -1.4644],
                [1.6393, 1.1556, -2.1791],
            ],
        )
        b4 = np.array([-0.0351, -0.0561, -0.0393])

        weights = [W1, W2, W3, W4]
        biases = [b1, b2, b3, b4]

        activ_funcs = [
            np.tanh,
            np.tanh,
            np.tanh,
        ]

        for w, b, f in zip(weights, biases, activ_funcs):

            # linear transform
            inner_array = inner_array @ w.T
            outer_array = outer_array @ w.T

            inner_anim = [
                dot.animate.move_to(ax.c2p(*pos))
                for dot, pos in zip(inner_dots, inner_array)
            ]
            outer_anim = [
                dot.animate.move_to(ax.c2p(*pos))
                for dot, pos in zip(outer_dots, outer_array)
            ]

            self.play(*inner_anim, *outer_anim, run_time=3, rate_func=linear)

            # translation
            inner_array = inner_array + b
            outer_array = outer_array + b

            inner_anim = [
                dot.animate.move_to(ax.c2p(*pos))
                for dot, pos in zip(inner_dots, inner_array)
            ]
            outer_anim = [
                dot.animate.move_to(ax.c2p(*pos))
                for dot, pos in zip(outer_dots, outer_array)
            ]

            self.play(*inner_anim, *outer_anim, run_time=3, rate_func=linear)

            # nonlinearity
            inner_array = f(inner_array)
            outer_array = f(outer_array)

            inner_anim = [
                dot.animate.move_to(ax.c2p(*pos))
                for dot, pos in zip(inner_dots, inner_array)
            ]
            outer_anim = [
                dot.animate.move_to(ax.c2p(*pos))
                for dot, pos in zip(outer_dots, outer_array)
            ]

            self.play(*inner_anim, *outer_anim, run_time=3, rate_func=linear)

        self.wait(2)


class SeparatingSpirals3dTraining(ThreeDScene):
    def construct(self):
        x_range = [-2.5, 2.5, 0.5]
        y_range = [-2.5, 2.5, 0.5]
        z_range = [-2.5, 2.5, 0.5]

        self.set_camera_orientation(phi=75 * DEGREES)

        # each axes here is represents a layer
        ax_grid = VGroup()
        for i in range(8):
            ax = ThreeDAxes(
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                x_length=10 / 3,
                y_length=10 / 3,
                z_length=10 / 3,
                tips=False,
                axis_config={"tick_size": 0.05},
            ).set_stroke(color=colors.BLACK)
            ax_grid.add(ax)
        ax_grid.arrange_in_grid(2, buff=0.05)

        t = np.arange(1, 5, 0.1)
        array1 = (np.array([np.cos(t), np.sin(t)]) * t).T * 0.5
        array2 = (np.array([np.cos(t + np.pi), np.sin(t + np.pi)]) * t).T * 0.5

        self.layers = []

        dots_all = Group()
        for ax in ax_grid:
            dots1 = Group(
                *[
                    Dot3D(ax.c2p(*point, 0), radius=0.04, color=colors.PURPLE)
                    for point in array1
                ]
            )

            dots2 = Group(
                *[
                    Dot3D([ax.c2p(*point, 0)], radius=0.04, color=colors.RED)
                    for point in array2
                ]
            )
            dots_all.add(Group(dots1, dots2))

        self.play(FadeIn(Group(ax_grid)))

        # setting up data for training
        labels1 = np.ones(len(array1)).reshape(-1, 1)
        labels2 = np.zeros(len(array2)).reshape(-1, 1)

        class1_data = np.concatenate(
            (array1.T, labels1.T, np.arange(len(array1)).reshape(1, -1)), axis=0
        )
        class2_data = np.concatenate(
            (array2.T, labels2.T, np.arange(len(array2)).reshape(1, -1)), axis=0
        )

        # consider adding a shuffling mechanism
        data = np.concatenate((class1_data, class2_data), axis=1)
        data = torch.tensor(data, dtype=torch.float32).T

        # setting up the model
        model = Model().float()

        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        model.requires_grad_(True)

        # getting the initial position
        x, labels, indices = (
            data[:, 0:2],
            data[:, 2].unsqueeze(1),
            data[:, 3].unsqueeze(1),
        )
        pred, hidden_out = model(x)
        with torch.no_grad():
            for ax, dots, layer_num in zip(ax_grid, dots_all, range(8)):
                layer = hidden_out[layer_num].numpy()
                for h_out, label, i in zip(layer, labels, indices):
                    if label == 0:
                        dots[0][int(i.item())].move_to(ax.c2p(*h_out))
                    else:
                        dots[1][int(i.item())].move_to(ax.c2p(*h_out))

        self.play(FadeIn(dots_all))

        # self.begin_ambient_camera_rotation(-0.1)

        # training loop
        for epoch in range(50):
            x, labels, indices = (
                data[:, 0:2],
                data[:, 2].unsqueeze(1),
                data[:, 3].unsqueeze(1),
            )
            pred, hidden_out = model(x)
            loss = F.binary_cross_entropy_with_logits(pred, labels)
            with torch.no_grad():
                animations = []
                for ax, dots, layer_num in zip(ax_grid, dots_all, range(8)):
                    layer = hidden_out[layer_num].numpy()
                    for h_out, label, i in zip(layer, labels, indices):
                        if label == 0:
                            animations.append(
                                dots[0][int(i.item())].animate.move_to(ax.c2p(*h_out))
                            )
                        else:
                            animations.append(
                                dots[1][int(i.item())].animate.move_to(ax.c2p(*h_out))
                            )

                if epoch % 1 == 0:
                    print(f"epoch: {epoch}, loss: {loss.item():.4f}")
            rotation_animations = [Rotate(ax, 0.0025 * TAU, UP) for ax in ax_grid]
            self.play(
                *rotation_animations, *animations, run_time=0.005, rate_func=linear
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.wait(2)
