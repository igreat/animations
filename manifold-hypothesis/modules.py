from manim import *
import colors
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_utils.layer import Layer
import random
from functools import partial
from dataclasses import dataclass


class Grid(VGroup):
    def __init__(self, axes, x_range, y_range, lattice_radius=0, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        self.submobjects = VGroup()
        self.grid_2d = []
        positions = []
        y_range[1] += y_range[2]
        x_range[1] += x_range[2]
        for y in np.arange(*y_range):
            row = []
            for x in np.arange(*x_range):
                point = Dot(axes.c2p(x, y), lattice_radius, color=colors.BLACK)
                row.append(point)
                self.submobjects.add(point)
                positions.append([x, y])

            self.grid_2d.append(row)

        self.grid_lines = VGroup()

        def update_grid_connection(m, pos1, pos2):
            x1, y1 = pos1
            x2, y2 = pos2
            return m.set_points_by_ends(
                self.grid_2d[y1][x1].get_center(),
                self.grid_2d[y2][x2].get_center(),
            )

        # building the horizontal lines
        for i in range(len(self.grid_2d[0])):
            for j in range(len(self.grid_2d) - 1):
                line = Line(color=colors.BLACK)
                update_line = partial(
                    update_grid_connection, pos1=[i, j], pos2=[i, j + 1]
                )
                line.add_updater(update_line)
                self.grid_lines.add(line)

        self.array = np.array(positions)
        # building the vertical lines
        for j in range(len(self.grid_2d)):
            for i in range(len(self.grid_2d[0]) - 1):
                line = Line(color=colors.BLACK)
                update_line = partial(
                    update_grid_connection, pos1=[i, j], pos2=[i + 1, j]
                )
                line.add_updater(update_line)
                self.grid_lines.add(line)


class DotsPlot(VGroup, metaclass=ConvertToOpenGL):
    # I initially used this to do reverse transformations, but I already gave up on that
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


class MnistImage:
    def __init__(self, image: np.ndarray, position: np.ndarray, height=0.5):
        self.image = ImageMobject(image)
        self.image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        self.image.height = height
        self.position_dot = Dot(
            position, radius=0
        )  # representing the position as a dot for convenience
        self.image.move_to(position)
        self.image.add_updater(lambda m, dt: m.move_to(self.position_dot))


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


class MnistClassifier(nn.Module):
    """
    purpose is to extract features (intermediate layers) of a simple mnist classifier
    """

    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        features = [x]
        x = self.fc1(x)
        features.append(x)
        x = torch.tanh(x)
        features.append(x)
        x = self.fc2(x)
        features.append(x)
        x = torch.tanh(x)
        features.append(x)
        x = self.fc3(x)
        features.append(x)
        return features


class VisualizationModel(nn.Module):
    def __init__(self, scene: Scene) -> None:
        super(VisualizationModel, self).__init__()

        # maybe turn the color of the nodes into a Value object
        input_length = 2
        self.input_nodes = VGroup(
            *[Circle(radius=0.25, color=colors.BLACK) for _ in range(input_length)]
        )
        self.input_nodes.arrange(DOWN).move_to([-6, 0, 0])

        self.fc1 = Layer(input_length, 8, scene)
        self.fc2 = Layer(self.fc1.layer.out_features, 8, scene)
        self.fc3 = Layer(self.fc2.layer.out_features, 8, scene)
        self.output_layer = Layer(self.fc2.layer.out_features, 1, scene)

        # consider moving all the bottom part into a super class
        self.layers = [
            self.fc1,
            self.fc2,
            self.fc3,
            self.output_layer,
        ]

        self.fc1.nodes.move_to([-3, 0, 0])
        self.fc2.nodes.move_to([0, 0, 0])
        self.fc3.nodes.move_to([3, 0, 0])
        self.output_layer.nodes.move_to([6, 0, 0])

        self.fc1.build_lines(self.input_nodes)
        self.fc2.build_lines(self.fc1.nodes)
        self.fc3.build_lines(self.fc2.nodes)
        self.output_layer.build_lines(self.fc3.nodes)

        self.fc1.lines.resume_updating()
        self.fc2.lines.resume_updating()
        self.fc3.lines.resume_updating()
        self.output_layer.lines.resume_updating()

        self.nodes = VGroup(
            self.input_nodes,
            self.fc1.nodes,
            self.fc2.nodes,
            self.fc3.nodes,
            self.output_layer.nodes,
        )
        self.lines = VGroup(
            self.fc1.lines,
            self.fc2.lines,
            self.fc3.lines,
            self.output_layer.lines,
        )
        self.mobs = VGroup(self.nodes, self.lines)

    def forward(self, x):
        outputs = []

        x = self.fc1.layer(x)
        x = torch.tanh(x)
        outputs.append(x)
        x = self.fc2.layer(x)
        x = torch.tanh(x)
        outputs.append(x)
        x = self.fc3.layer(x)
        x = torch.tanh(x)
        outputs.append(x)
        x = self.output_layer.layer(x)
        outputs.append(torch.tanh(x.detach()))

        with torch.no_grad():
            # TODO: consider making this functionality inside each layer
            #       where each layer has a "value" attribute that maps to the nodes

            # the activations shown are of a random point in the batch
            point = random.randint(0, len(x) - 1)
            for layer, opacity in zip(self.layers, outputs):
                # I will take the mean across the batch dimention
                opc = opacity.squeeze().detach()
                if opc.dim() == 1:
                    opc = [opc[point].item()]
                else:
                    opc = opc[point].tolist()

                layer.node_opacities = opc
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.layer.parameters()
        return params

    def reset_colors(self):
        for layer in self.layers:
            layer.node_opacities = [0] * len(layer.nodes)
