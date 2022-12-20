from manim import *
import torch
import colors
import numpy as np
from utils import generate_outer_inner_circles
from models import DiskClassifier2D
from modules import Grid

config.background_color = colors.WHITE

# TODO: when doing the transformation, maybe show the code of the 
#       neural network on the side

class SeparatingDisks2D(Scene):
    def construct(self):
        self.separate_disks()
        self.wait()

    def separate_disks(self):

        # TODO: when doing the transformation, maybe show the code of the
        #       neural network on the side


        # load the disk classifier model
        diskclassifier = DiskClassifier2D()
        diskclassifier.load_state_dict(
            torch.load("saved_models/separating_disks2d_tanh.pt")
        )

        # setting up the axes
        x_range = [-3.5, 3.5, 0.5]
        y_range = [-3.5, 3.5, 0.5]

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

        # generate the inner and outer dots
        (inner_dots, inner_array), (
            outer_dots,
            outer_array,
        ) = generate_outer_inner_circles(ax, 100, dot_constructor=Dot)

        # create the grid
        grid = Grid(ax, [-3.5, 3.5, 0.25], [-3.5, 3.5, 0.25])

        grid.set_color(colors.BLACK)
        grid.grid_lines.set_stroke(width=0.5)

        self.play(Write(ax))
        self.play(Write(grid.grid_lines))
        self.play(Write(VGroup(*inner_dots, *outer_dots)))
        self.wait()

        # get the intermediate outputs of the inner, outer and grid arrays
        outer_outputs = diskclassifier(torch.tensor(outer_array).float())
        inner_outputs = diskclassifier(torch.tensor(inner_array).float())
        grid_outputs = diskclassifier(torch.tensor(grid.array).float())

        # animate the transformation
        for inner_output, outer_output, grid_output in zip(
            inner_outputs[:-1], outer_outputs[:-1], grid_outputs[:-1]
        ):
            # Note: we normalize the outputs to a range of [-3.5, 3.5] 
            #       so that they aren't blown out of proportion

            shift = -grid_output.min()
            reduce_factor = grid_output.max() - grid_output.min()

            # normalizing the outputs
            inner_output = (inner_output + shift) / reduce_factor
            outer_output = (outer_output + shift) / reduce_factor
            grid_output = (grid_output + shift) / reduce_factor


            # now making the range -3.5 to 3.5
            inner_output = ((inner_output - 0.5) * 2) * 3.5
            outer_output = ((outer_output - 0.5) * 2) * 3.5
            grid_output = ((grid_output - 0.5) * 2) * 3.5

            new_inner = [
                dot.animate.move_to(ax.c2p(*pos))
                for dot, pos in zip(inner_dots, inner_output.detach())
            ]
            new_outer = [
                dot.animate.move_to(ax.c2p(*pos))
                for dot, pos in zip(outer_dots, outer_output.detach())
            ]
            new_grid = [
                dot.animate.move_to(ax.c2p(*pos))
                for dot, pos in zip(grid.submobjects, grid_output.detach())
            ]

            self.play(
                *new_inner,
                *new_outer,
                *new_grid,
                run_time=3,
                rate_func=rate_functions.linear,
            )

        self.wait()
        