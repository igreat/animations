from manim import *
import torch
import colors
import numpy as np
from utils import generate_outer_inner_circles
from models import DiskClassifier3D

config.background_color = colors.WHITE


class SeparatingDisks3D(ThreeDScene):
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
        z_range = [-2.5, 2.5, 0.5]

        ax = ThreeDAxes(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            x_length=7,
            y_length=7,
            z_length=7,
            tips=False,
        ).set_stroke(color=colors.BLACK)

        # generate the inner and outer dots
        (inner_dots, inner_array), (
            outer_dots,
            outer_array,
        ) = generate_outer_inner_circles(ax, 60)

        self.begin_ambient_camera_rotation(-0.1)

        self.play(Write(ax))
        self.play(FadeIn(inner_dots))
        self.play(FadeIn(outer_dots))

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

        # get the normal vector of the plane (equivalent to the last layer's weights)
        final_weight = list(diskclassifier.parameters())[-2][0].detach().numpy()
        normal_vector = final_weight / np.linalg.norm(final_weight)

        # getting the polar coordinates of the normal vector
        theta = np.arccos(normal_vector[2])
        phi = np.arctan2(normal_vector[1], normal_vector[0])

        # the distance of the plane from the origin
        bias = list(diskclassifier.parameters())[-1][0].detach().numpy()
        distance = bias / np.linalg.norm(final_weight)

        normal_arrow = Arrow(
            ax.c2p(*ORIGIN),
            ax.c2p(*(normal_vector * 0.5)),
            color=colors.DARK_RED,
            buff=0,
        ).shift(ax.c2p(*(-normal_vector * distance)))

        ### separating the two regions with a plane ###

        # define a hyperplane as a surface object: (a compromise for the hyperplane bug)
        # this still appears in front of everything but has a built in square lattice pattern
        # which allows me to make the opacity very low
        hyperplane = (
            Surface(
                lambda u, v: ax.c2p(u, v, 0), u_range=[-2.5, 2.5], v_range=[-2.5, 2.5]
            )
            .set_fill(color=colors.DESERT, opacity=0.75)
            .move_to(ax.c2p(0, 0, 0))
            .rotate(PI / 2, axis=ax.c2p(*UP))
            .set_stroke(width=0)
        )

        # rotate the plane to have a normal vector of normal_vector
        # wait, why did I make the about_point not the origin here?
        hyperplane.rotate(angle=theta, axis=ax.c2p(*IN), about_point=ax.c2p(*ORIGIN))
        hyperplane.rotate(angle=phi, axis=ax.c2p(*RIGHT), about_point=ax.c2p(*ORIGIN))

        # shifting the plane to the correct distance from the origin
        hyperplane.move_to(ax.c2p(*(-normal_vector * distance)))

        self.play(FadeIn(hyperplane), FadeIn(normal_arrow))
        self.wait(35)
