from manim import *
import colors
import numpy as np
from models import SpiralsClassifier2D
import torch
from torch import optim
import torch.nn.functional as F

config.background_color = colors.WHITE


class SeparatingSpirals2dTraining(Scene):
    def construct(self):
        x_range = [-2.5, 2.5, 0.5]
        y_range = [-2.5, 2.5, 0.5]

        # here what I actually want is multiple axis arranged in a grid (use .arrange(rows, cols))
        # each grid will be for a particular layer

        # ALSO SHOW HOW THE BOUNDARY CHANGES WITH TIME!

        # TODO: show how the final separation line evolves with time

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
        # something could go wrong here since I apparently used to call this TrainingModel...
        model = SpiralsClassifier2D().float()

        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        model.requires_grad_(True)

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

        # training loop
        for epoch in range(200):
            x, labels, indices = (
                data[:, 0:2],
                data[:, 2].unsqueeze(1),
                data[:, 3].unsqueeze(1),
            )
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
            self.play(*animations, run_time=0.005, rate_func=linear)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.wait(2)
