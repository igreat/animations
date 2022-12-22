from manim import *
from utils import colors
import numpy as np
from utils.models import SpiralsClassifier2D
import torch
from torch import optim
import torch.nn.functional as F

config.background_color = colors.WHITE


class SeparatingSpirals2dTraining(Scene):
    def construct(self):
        x_range = [-2.5, 2.5, 0.5]
        y_range = [-2.5, 2.5, 0.5]

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

        surrounding_boxes = VGroup()
        for ax in ax_grid:
            surrounding_boxes.add(
                SurroundingRectangle(
                    ax,
                    buff=0,
                    color=colors.BLACK,
                ).set_stroke(width=1.5)
            )

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

        self.play(Write(surrounding_boxes))
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
        # something could go wrong here since I apparently used to call this TrainingModel...
        model = SpiralsClassifier2D().float()

        # getting the initial position
        x, labels, indices = (
            data[:, 0:2],
            data[:, 2].unsqueeze(1),
            data[:, 3].unsqueeze(1),
        )
        outputs = model(x)
        pred, hidden_out = outputs[-1], outputs[0:-1]
        with torch.no_grad():
            for ax, dots, layer_num in zip(ax_grid, dots_all, range(8)):
                layer = hidden_out[layer_num].numpy()
                for h_out, label, i in zip(layer, labels, indices):
                    if label == 0:
                        dots[0][int(i.item())].move_to(ax.c2p(*h_out))
                    else:
                        dots[1][int(i.item())].move_to(ax.c2p(*h_out))

        self.play(Write(dots_all))

        def get_line() -> Line:
            # doing it through rotation and shifting for an easy way to maintain the line's length

            # get the normal vector of the line (equivalent to the last layer's weights)
            final_weight = list(model.parameters())[-2][0].detach().numpy()
            normal_vector = final_weight / np.linalg.norm(final_weight)

            # getting the polar coordinates of the normal vector
            theta = np.arctan2(normal_vector[1], normal_vector[0])

            # the distance of the line from the origin
            bias = list(model.parameters())[-1][0].detach().numpy()
            distance = bias / np.linalg.norm(final_weight)

            # define the line
            line = Line(DOWN * 1.25, UP * 1.25).set_stroke(color=colors.BLACK)

            # rotate the line to the normal vector
            line.rotate(theta).move_to(ax_grid[-1].c2p(*ORIGIN))

            # shift the line to the correct distance from the origin
            line.move_to(ax_grid[-1].c2p(*(-normal_vector * distance)))

            return line

        line = always_redraw(get_line)

        self.play(Write(line))

        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        model.requires_grad_(True)

        # training loop
        for epoch in range(300):
            outputs = model(x)
            pred, hidden_out = outputs[-1], outputs[0:-1]
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

            self.play(*animations, run_time=0.01, rate_func=linear)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.wait(2)
