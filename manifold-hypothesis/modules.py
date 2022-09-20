from manim import *
import colors
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
import torch
from torch import nn
import torch.nn.functional as F


class Grid(VGroup):
    def __init__(self, axes, x_range, y_range, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        self.submobjects = VGroup()
        self.grid_2d = []
        positions = []
        y_range[1] += y_range[2]
        x_range[1] += x_range[2]
        for y in np.arange(*y_range):
            row = []
            for x in np.arange(*x_range):
                point = Dot(axes.c2p(x, y), 0)
                row.append(point)
                self.submobjects.add(point)
                positions.append([x, y])

            self.grid_2d.append(row)

        self.grid_lines = VGroup()

        for i in range(len(self.grid_2d[0])):
            for j in range(len(self.grid_2d[1]) - 1):

                line = Line(color=colors.BLACK)
                line.add_updater(
                    lambda m, dt, i=i, j=j: m.set_points_by_ends(
                        self.grid_2d[i][j].get_center(),
                        self.grid_2d[i][j + 1].get_center(),
                    )
                )
                self.grid_lines.add(line)

        self.array = np.array(positions)

        for j in range(len(self.grid_2d[1])):
            for i in range(len(self.grid_2d[0]) - 1):

                line = Line(color=colors.BLACK)
                line.add_updater(
                    lambda m, dt, i=i, j=j: m.set_points_by_ends(
                        self.grid_2d[i][j].get_center(),
                        self.grid_2d[i + 1][j].get_center(),
                    )
                )
                self.grid_lines.add(line)


class DotsPlot(VGroup, metaclass=ConvertToOpenGL):
    def __init__(self, ax, func, start, end, num_points, *vmobjects, **kwargs):
        VGroup.__init__(self, **kwargs)

        xs = np.linspace(start, end, num_points)
        ys = func(xs)

        self.submobjects = VGroup(*[Dot(ax.c2p(x, y), 0) for x, y in zip(xs, ys)])

        self.sublines = VGroup()

        for i in range(0, len(self.submobjects) - 1):
            line = Line(color=colors.BLACK)
            line.add_updater(
                lambda m, dt, i=i, j=i + 1: m.set_points_by_ends(
                    self.submobjects[i].get_center(),
                    self.submobjects[j].get_center(),
                )
            )
            line.resume_updating()
            self.sublines.add(line)

        self.array = np.concatenate((xs.reshape(-1, 1), ys.reshape(-1, 1)), axis=1)


class Spirals2dModel(nn.Module):
    def __init__(self) -> None:
        super(Spirals2dModel, self).__init__()

        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)
        self.fc4 = nn.Linear(2, 2)
        self.fc5 = nn.Linear(2, 1)

    def forward(self, input):
        outputs = [input]

        x = self.fc1(input)
        outputs.append(x)
        x = torch.tanh(x)
        outputs.append(x)

        x = self.fc2(x)
        outputs.append(x)
        x = torch.tanh(x)
        outputs.append(x)

        x = self.fc3(x)
        outputs.append(x)
        x = torch.tanh(x)
        outputs.append(x)

        x = self.fc4(x)
        outputs.append(x)
        x = torch.tanh(x)
        outputs.append(x)

        x = self.fc5(x)
        outputs.append(x)

        return outputs


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 3)
        self.fc4 = nn.Linear(3, 3)
        self.fc5 = nn.Linear(3, 1)

    def forward(self, input):
        outputs = [input]

        x = self.fc1(input)
        outputs.append(x)
        x = torch.tanh(x)
        outputs.append(x)

        x = self.fc2(x)
        outputs.append(x)
        x = torch.tanh(x)
        outputs.append(x)

        x = self.fc3(x)
        outputs.append(x)
        x = torch.tanh(x)
        outputs.append(x)

        x = self.fc4(x)
        outputs.append(x)
        x = torch.tanh(x)
        outputs.append(x)

        return self.fc5(x), outputs
