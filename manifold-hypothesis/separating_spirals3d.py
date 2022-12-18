from manim import *
import colors
import numpy as np
from utils import *
from modules import *
from models import DiskClassifier3D

config.background_color = colors.WHITE


class SeparatingSpirals3d(ThreeDScene):
    def construct(self):
        self.separate_disks()
        self.wait()

    def separate_disks(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        # load the disk classifier model
        diskclassifier = DiskClassifier3D()
        diskclassifier.load_state_dict(
            torch.load("saved_models/separating_disks3d_tanh.pt")
        )

        # setting up the axes
        x_range = [-2.5, 2.5, 0.5]
        y_range = [-2.5, 2.5, 0.5]

        ax = ThreeDAxes(
            x_range=x_range,
            y_range=y_range,
            x_length=7,
            y_length=7,
            tips=False,
        ).set_stroke(color=colors.BLACK)

        # generate the inner and outer dots
        (inner_dots, inner_array), (
            outer_dots,
            outer_array,
        ) = generate_outer_inner_circles(ax, 100)

        self.begin_ambient_camera_rotation(-0.1)

        print(inner_array.shape, outer_array.shape)
        self.play(Write(ax))
        self.play(Write(inner_dots))
        self.play(Write(outer_dots))

        # get the intermediate outputs of the inner and outer arrays
        inner_outputs = diskclassifier(torch.tensor(inner_array).float())
        outer_outputs = diskclassifier(torch.tensor(outer_array).float())

        # animate the inner and outer arrays
        for inner_output, outer_output in zip(inner_outputs[:-1], outer_outputs[:-1]):
            new_inner_dots = [
                dot.animate.move_to(ax.c2p(*pos))
                for dot, pos in zip(inner_dots, inner_output.detach())
            ]
            new_outer_dots = [
                dot.animate.move_to(ax.c2p(*pos))
                for dot, pos in zip(outer_dots, outer_output.detach())
            ]

            self.play(
                *new_inner_dots,
                *new_outer_dots,
                run_time=3,
                rate_func=rate_functions.linear,
            )

        self.wait()

        ### separating it with a plane ###
        # TODO: use the last layer's weights to create the separating plane
        hyperplane = (
            Square(ax.x_length)
            .set_fill(color=colors.DESERT, opacity=0.5)
            .set_stroke(color=colors.DESERT, width=4)
            .move_to(ax.get_center())
        )
        # hyperplane good so far

        self.play(Create(hyperplane))
        self.wait()
