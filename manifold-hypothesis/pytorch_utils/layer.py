from torch import nn
from manim import *
from utils import colors


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

        # TODO: consider turning this into an array of Values
        self.node_opacities = [0] * output_dim

        self.nodes = VGroup(
            *[
                Circle(radius=0.25)
                .set_stroke(color=colors.BLACK)
                .set_fill(color=colors.BLACK, opacity=opacity)
                for opacity in self.node_opacities
            ]
        )

        def update_color(m, dt, i):
            clr = colors.GREEN
            opac = self.node_opacities[i]

            if opac < 0:
                clr = colors.DARK_RED
                opac *= -1
            m.set_fill(color=clr, opacity=opac)

        for i, node in enumerate(self.nodes):
            # this is a weird way of doing it, ask about how not to have to do it like this!
            node.add_updater(lambda m, dt, i=i: update_color(m, dt, i))
            node.resume_updating()

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

    def get_layer_mobs(self):
        return VGroup(self.lines, self.nodes, self.labels)


def build_layer_lines(
    layer1: VGroup,
    layer2: VGroup,
    opacities,
    colors=[GREEN, RED],
    dotted=False,
    stuck=True,
    start=ORIGIN,
    end=ORIGIN,
    always_back=False,
    inflate_opacities=1,
    width=4,
) -> VGroup:
    # this builds a single line between two mobjects

    # colors should only have two elements

    def get_layer_lines():
        connection_lines = VGroup()
        i = 0
        for mob1 in layer1:
            point1 = mob1.get_critical_point(start)
            for mob2 in layer2:
                point2 = mob2.get_critical_point(end)
                # print(colors[i].get_center())
                opacity = opacities[i]
                if type(opacity) == ValueTracker:
                    opacity = opacity.get_value()

                if opacity > 0:
                    color = colors[0]
                else:
                    color = colors[1]

                if dotted:
                    connection_lines.add(
                        DashedLine(point1, point2, color=color).set_stroke(
                            width=width,
                            opacity=clip(abs(inflate_opacities * opacity), 0, 1),
                        )
                    )
                else:
                    connection_lines.add(
                        Line(point1, point2, color=color).set_stroke(
                            width=width,
                            opacity=clip(abs(inflate_opacities * opacity), 0, 1),
                        )
                    )
                i += 1
        if always_back:
            connection_lines.z_index = 0
        return connection_lines

    if stuck:
        return always_redraw(get_layer_lines)

    return get_layer_lines()
