from torch import nn
from manim import *
import colors
from video_utils import *


def build_labels(nodes, layer_index):
    labels = VGroup()
    for i in range(len(nodes)):
        label = MathTex(
            r"a_{" + str(i) + r"}^{" + str(layer_index) + r"}",
            font_size=30,
            color=colors.BLACK,
        ).set_stroke(width=1)
        label.add_updater(lambda m, i=i: m.next_to(nodes[i], ORIGIN))
        label.resume_updating()
        labels.add(label)

    return labels


class Layer:
    def __init__(
        self,
        input_dim,
        output_dim,
        scene: Scene,
        input_displayed=None,
    ) -> None:

        if input_displayed is None:
            input_displayed = input_dim

        self.layer = nn.Linear(input_dim, output_dim)
        self.weights = [ValueTracker() for _ in range(input_dim * output_dim)]
        self.biases = [ValueTracker() for _ in range(output_dim)]
        # this updater thing is very possibly not going to work
        # be ready to remove it

        def match_weights(dt=1):
            i = 0
            for layer in self.layer.weight:
                for weight in layer[:input_displayed]:
                    self.weights[i].set_value(weight.item())
                    i += 1

        def match_biases(dt=1):
            i = 0
            for bias in self.layer.bias[:input_displayed]:
                self.biases[i].set_value(bias.item())
                i += 1

        self.nodes = VGroup(
            *[Circle(radius=0.25, color=colors.BLACK) for _ in range(output_dim)]
        )
        self.nodes.arrange(DOWN)

        match_weights()
        match_biases()
        scene.add_updater(match_weights)
        scene.add_updater(match_biases)

    def build_lines(
        self,
        input_nodes,
        colors=[colors.GREEN, colors.RED],
        start=RIGHT,
        end=LEFT,
        inflate_opacities=3,
    ):
        # builds the lines between the previous nodes and current nodes
        self.lines = build_layer_lines(
            input_nodes,
            self.nodes,
            self.weights,
            colors,
            start=start,
            end=end,
            inflate_opacities=inflate_opacities,
        )
        return self

    def build_labels(self, layer_index):
        self.labels = build_labels(self.nodes, layer_index)
        return self