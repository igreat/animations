from manim import *
import colors
import numpy as np
from utils import *
from modules import *

config.background_color = colors.WHITE


class SeparatingSpirals3d(ThreeDScene):
    def construct(self):
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
