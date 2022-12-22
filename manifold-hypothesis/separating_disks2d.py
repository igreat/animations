from manim import *
import torch
from utils import colors
from utils.other_utils import generate_outer_inner_circles
from utils.models import DiskClassifier2D
from utils.modules import Grid

config.background_color = colors.WHITE


class SeparatingDisks2D(Scene):
    def construct(self):
        self.separate_disks()
        self.wait()

    def separate_disks(self):

        #### SETTING UP THE SCENE ####

        code = r"""import torch
from torch import nn

class DiskClassifier2D(nn.Module):
    def __init__(self) -> None:
        super(DiskClassifier2D, self).__init__()
        # fully connected linear layers (FC)
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)
        self.fc4 = nn.Linear(2, 2)
        self.fc5 = nn.Linear(2, 1)

    def forward(self, input):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return self.fc5(x)
"""
        model_code = Code(
            code=code,
            tab_width=2,
            background="rectangle",
            language="Python",
            font="Menlo, Monaco",
            style="vs",
            background_stroke_color=colors.BLACK,
            background_stroke_width=4,
            line_spacing=1,
            font_size=18,
        )

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

        model_code.height = ax.height * 0.75

        # generate the inner and outer dots
        (inner_dots, inner_array), (
            outer_dots,
            outer_array,
        ) = generate_outer_inner_circles(ax, 150, dot_constructor=Dot)

        # create the grid
        grid = Grid(ax, [-3.5, 3.5, 0.25], [-3.5, 3.5, 0.25])

        grid.set_color(colors.BLACK)
        grid.grid_lines.set_stroke(width=0.5)

        self.play(Write(ax))
        self.play(Write(grid.grid_lines))
        self.play(Write(VGroup(*inner_dots, *outer_dots)))
        self.wait()

        self.play(Write(model_code))
        self.wait()

        full_graph = Group(ax, grid.submobjects, inner_dots, outer_dots)
        self.play(Group(full_graph, model_code).animate.arrange(RIGHT, buff=1))
        self.wait()


        #### TRANSFORMING THE DISKS ####

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
        