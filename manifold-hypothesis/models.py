from manim import *
import torch
from torch import nn
import random
from pytorch_utils.layer import Layer
import colors


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
