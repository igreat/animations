from manim import *
from utils import colors
import numpy as np
from utils.other_utils import add_brackets
from utils.modules import Grid
from utils.models import SpiralsClassifier2D, VisualizationModel
import torch
from torch import optim
import torch.nn.functional as F
from pytorch_utils.layer import build_layer_lines


config.background_color = colors.WHITE

# why doesn't \cross work?
mul_tex = MathTex(
    r"\dot",
    color=colors.BLACK,
    font_size=25,
).set_stroke(width=1)

plus_tex = MathTex(
    r"+",
    color=colors.BLACK,
    font_size=25,
).set_stroke(width=1)

equals_tex = MathTex(
    r"=",
    color=colors.BLACK,
    font_size=25,
).set_stroke(width=1)


class SeparatingSpirals2d(Scene):
    def construct(self):

        self.build_spirals()
        self.wait()

        self.model = SpiralsClassifier2D()
        self.model.load_state_dict(
            torch.load("saved_models/separating_spirals2d_tanh.pt")
        )
        self.model.eval()

        self.show_neural_network()
        self.wait()
        self.show_boundary()
        self.wait()
        self.train_neural_net()
        self.wait()
        self.show_mapping()
        self.wait()

        self.transform()
        self.wait(2)

    def build_spirals(self):
        self.x_range = [ValueTracker(-3.5), ValueTracker(3.5), ValueTracker(0.5)]
        self.y_range = [ValueTracker(-3.5), ValueTracker(3.5), ValueTracker(0.5)]

        def build_axes(x_range, y_range):

            if type(x_range[0]) == ValueTracker:
                x_range = [i.get_value() for i in x_range]

            if x_range == [0, 0, 0]:
                # in case the axis collapses into a point
                x_range = [0, 1, 0.25]

            if type(y_range[0]) == ValueTracker:
                y_range = [i.get_value() for i in y_range]

            if y_range == [0, 0, 0]:
                # in case the axis collapses into a point
                y_range = [0, 1, 0.25]

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

            return ax

        self.ax = build_axes(self.x_range, self.y_range)
        # ax = always_redraw(lambda: build_axes(self.x_range, self.y_range))

        t = np.arange(1, 5, 0.05)
        self.blue_spiral_array = (np.array([np.cos(t), np.sin(t)]) * t).T * 0.5
        self.red_spiral_array = (
            np.array([np.cos(t + np.pi), np.sin(t + np.pi)]) * t
        ).T * 0.5

        # t = np.arange(1, 11, 0.1) * 0.4
        # self.blue_spiral_array = (np.array([np.cos(t), np.sin(t)]) * t).T
        # self.red_spiral_array = (np.array([np.cos(t + np.pi), np.sin(t + np.pi)]) * t).T

        self.blue_spiral = VGroup(
            *[
                Dot(self.ax.c2p(*point), radius=0.05, color=colors.PURPLE)
                for point in self.blue_spiral_array
            ]
        )

        self.red_spiral = VGroup(
            *[
                Dot(self.ax.c2p(*point), radius=0.05, color=colors.RED)
                for point in self.red_spiral_array
            ]
        )

        self.spiral_mobs = VGroup(self.ax, self.red_spiral, self.blue_spiral)
        self.play(Write(self.ax))
        # self.play(Write(grid.grid_lines))
        self.play(Write(self.red_spiral))
        self.play(Write(self.blue_spiral))
        self.wait()
        self.play(self.spiral_mobs.animate.move_to([4.5, 0, 0]).scale(0.6))

    def show_neural_network(self):

        self.vis_model = VisualizationModel(scene=self)
        self.vis_model.mobs.move_to([-2.5, 0, 0]).scale(0.7)
        self.play(
            Write(self.vis_model.nodes),
            Write(self.vis_model.lines),
        )
        self.wait()

    def show_boundary(self):

        points = []
        x_min, x_max, y_min, y_max = -3, 3, -3, 3
        x_num, y_num = 50, 50

        self.boundary_width = (x_max - x_min) / x_num
        self.boundary_height = (y_max - y_min) / y_num

        for x in np.linspace(x_min, x_max, x_num):
            for y in np.linspace(y_min, y_max, y_num):
                points.append([x, y])

        self.sample_points = torch.tensor(points)

        output = self.vis_model(torch.tensor(self.sample_points).float())
        final_output = torch.sigmoid(output)
        final_output = final_output >= 0.5

        # transparent squares
        self.boundary_rects = VGroup()
        for point, output in zip(self.sample_points, final_output):
            color = colors.PURPLE if output else colors.RED
            self.boundary_rects.add(
                Rectangle(height=self.boundary_height, width=self.boundary_width)
                .set_fill(color, 0.25)
                .set_stroke(width=0, opacity=0)
                .move_to(self.ax.c2p(*point))
            )

        self.bring_to_back(self.boundary_rects)
        self.play(FadeIn(self.boundary_rects))
        self.wait()

    def train_neural_net(self):
        # preparing data for training
        labels_blue = np.ones(len(self.blue_spiral_array)).reshape(-1, 1)
        labels_red = np.zeros(len(self.blue_spiral_array)).reshape(-1, 1)

        blue_spiral_l = np.concatenate((self.blue_spiral_array, labels_blue), axis=1)
        red_spiral_l = np.concatenate((self.red_spiral_array, labels_red), axis=1)

        data = np.concatenate((blue_spiral_l, red_spiral_l), axis=0)
        data = torch.tensor(data, dtype=torch.float32)

        optimizer = optim.Adam(self.vis_model.parameters(), lr=1e-2)
        self.vis_model.requires_grad_(True)

        for epoch in range(250):
            x, labels = data[:, 0:2], data[:, 2].unsqueeze(1)
            pred = self.vis_model(x)

            # updating boundary
            with torch.no_grad():
                output = self.vis_model(self.sample_points.float())
                final_output = torch.sigmoid(output)
                final_output = final_output >= 0.5
                # transparent squares color change
                animations = []
                for i, output in enumerate(final_output):
                    color = colors.PURPLE if output else colors.RED
                    animations.append(
                        self.boundary_rects[i].animate.set_fill(color=color)
                    )

                self.play(
                    *animations,
                    run_time=0.01,
                    rate_func=rate_functions.linear,
                )

            loss = F.binary_cross_entropy_with_logits(pred, labels)
            print_loss_interval = 10
            if epoch % print_loss_interval == 0:
                print(f"epoch: {epoch}, loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.wait()

    def show_mapping(self):
        self.play(FadeOut(self.boundary_rects))
        self.wait()

        self.play(
            FadeOut(self.vis_model.nodes[2:], shift=UP),
            FadeOut(self.vis_model.lines[1:], shift=UP),
        )

        self.play(
            self.vis_model.nodes[:2]
            .animate.arrange(RIGHT, buff=3)
            .scale(1.5)
            .move_to([-2.5, 0, 0])
        )
        self.wait()

        new_nodes = self.vis_model.nodes[0].copy()
        new_nodes.move_to(self.vis_model.nodes[1].get_center())

        self.play(Transform(self.vis_model.nodes[1], new_nodes))
        self.remove(self.vis_model.nodes[1])
        self.add(new_nodes)
        self.vis_model.nodes[1] = new_nodes

        new_lines = build_layer_lines(
            self.vis_model.input_nodes,
            self.vis_model.nodes[1],
            [1, 0.5, -0.5, -1],
            colors=[colors.GREEN, colors.RED],
            start=RIGHT,
            end=LEFT,
            inflate_opacities=1,
        )

        self.play(Transform(self.vis_model.lines[0], new_lines))
        self.remove(self.vis_model.lines[0])
        self.add(new_lines)
        self.vis_model.lines[0] = new_lines

        self.vis_model.reset_colors()
        self.play(
            self.vis_model.nodes[0].animate.scale(1.5),
            self.vis_model.nodes[1].animate.scale(1.5),
        )
        self.wait()

        def get_dot_pos(t):
            return np.array([np.cos(t), np.sin(t)]) * t

        t = ValueTracker(1)
        initial_point = [self.ax.c2p(*get_dot_pos(t.get_value()), 0)]

        tracked_dot = Dot(point=initial_point, radius=0.05, color=colors.BLACK)
        tracked_dot.add_updater(
            lambda x: x.move_to(self.ax.c2p(*get_dot_pos(t.get_value()), 0))
        )
        tracked_dot.z_index = 3

        def get_dot_text():
            x, y = get_dot_pos(t.get_value())

            x_text = (
                MathTex(
                    f"{x:.2f}",
                    color=colors.BLACK,
                    font_size=25,
                )
                .next_to(self.vis_model.input_nodes[0], ORIGIN)
                .set_stroke(width=1)
            )

            y_text = (
                MathTex(
                    f"{y:.2f}",
                    color=colors.BLACK,
                    font_size=25,
                )
                .next_to(self.vis_model.input_nodes[1], ORIGIN)
                .set_stroke(width=1)
            )

            return VGroup(x_text, y_text)

        dot_text = always_redraw(get_dot_text)

        self.play(Write(dot_text))
        self.play(t.animate.set_value(4.4), run_time=3)
        self.play(t.animate.set_value(1.0), run_time=3)
        self.wait()

        self.play(Write(tracked_dot))
        self.play(t.animate.set_value(4.4), run_time=3)
        self.play(t.animate.set_value(1.0), run_time=3)
        self.wait()

        def get_trans_pos(p):
            A = np.array([[0.25, 0.5], [0.5, 0.25]])
            return np.tanh(A @ p + [1, 1])

        trans_dot = Dot(point=initial_point, radius=0.05, color=colors.GRAY)
        trans_dot.add_updater(
            lambda x: x.move_to(
                self.ax.c2p(*get_trans_pos(get_dot_pos(t.get_value())), 0)
            )
        )

        def get_trans_text():
            x, y = get_trans_pos(get_dot_pos(t.get_value()))

            x_text = (
                MathTex(
                    f"{x:.2f}",
                    color=colors.BLACK,
                    font_size=25,
                )
                .next_to(self.vis_model.nodes[1][0], ORIGIN)
                .set_stroke(width=1)
            )

            y_text = (
                MathTex(
                    f"{y:.2f}",
                    color=colors.BLACK,
                    font_size=25,
                )
                .next_to(self.vis_model.nodes[1][-1], ORIGIN)
                .set_stroke(width=1)
            )

            return VGroup(x_text, y_text)

        trans_text = always_redraw(get_trans_text)
        trans_dot.z_index = 3
        self.play(Write(trans_text))
        self.play(Write(trans_dot))
        self.play(t.animate.set_value(4.4), run_time=3)
        self.play(t.animate.set_value(1.0), run_time=3)
        self.wait()

        highlight_lines1 = build_layer_lines(
            self.vis_model.input_nodes,
            self.vis_model.nodes[1][0],
            [1, -0.5],
            colors=[colors.GREEN, colors.RED],
            start=RIGHT,
            end=LEFT,
            inflate_opacities=1,
            stuck=False,
            width=8,
        )

        all_neurons = VGroup(
            self.vis_model.nodes[:2],
            self.vis_model.lines[0],
            highlight_lines1,
            trans_text,
            dot_text,
        )

        # highlighting input neurons and first output neuron
        self.play(
            *[
                node.animate.set_stroke(colors.ORANGE, 6).set_fill(colors.DESERT, 0.5)
                for node in [self.vis_model.nodes[1][0], *self.vis_model.nodes[0]]
            ],
            FadeIn(highlight_lines1),
        )
        self.wait()

        self.play(all_neurons.animate.shift(DOWN))
        self.wait()

        x_text1, y_text1 = get_dot_text()
        output_x, output_y = get_trans_text()

        self.add(x_text1, y_text1, output_x, output_y)

        self.play(x_text1.animate.shift(2 * UP + RIGHT * 0.8))

        mul_x_x = mul_tex.copy()
        mul_x_x.next_to(x_text1, RIGHT, 0.1)
        # TODO: review the notation to make sure it's consistent with convention
        w_x_x = (
            MathTex(
                r"w_{x}^{x}",
                color=colors.GREEN,
                font_size=25,
            )
            .next_to(mul_x_x, RIGHT, 0.1)
            .set_stroke(width=1)
        )

        self.play(Write(mul_x_x))
        self.play(Transform(highlight_lines1[0], w_x_x))
        self.remove(highlight_lines1[0])
        self.add(w_x_x)

        plus_x = plus_tex.copy()
        plus_x.next_to(w_x_x, RIGHT)
        self.play(Write(plus_x))

        self.play(y_text1.animate.next_to(plus_x, RIGHT))

        mul_x_y = mul_tex.copy()
        mul_x_y.next_to(y_text1, RIGHT, 0.1)
        self.play(Write(mul_x_y))

        w_x_y = (
            MathTex(
                r"w_{x}^{y}",
                color=colors.RED,
                font_size=25,
            )
            .next_to(mul_x_y, RIGHT, 0.1)
            .set_stroke(width=1)
        )
        self.play(Transform(highlight_lines1[1], w_x_y))
        self.remove(highlight_lines1[1])
        self.add(w_x_y)

        # consider here just introducing tanh right away and then just denote it as tanh()
        non_lin_wrapper1 = VGroup()
        non_lin_wrapper1.add(
            MathTex(r"g(", color=colors.BLACK, font_size=25).next_to(
                x_text1, LEFT, 0.15
            )
        )
        non_lin_wrapper1.add(
            MathTex(r")", color=colors.BLACK, font_size=25).next_to(w_x_y, RIGHT, 0.15)
        )
        non_lin_wrapper1.set_stroke(width=1)
        self.play(Write(non_lin_wrapper1))

        equal_x = equals_tex.copy()
        equal_x.next_to(non_lin_wrapper1, RIGHT)
        self.play(Write(equal_x))

        self.play(output_x.animate.next_to(equal_x, RIGHT))

        self.wait()

        # unhighlighting input neurons and first output neuron
        self.play(
            *[
                node.animate.set_stroke(colors.BLACK, 4).set_fill(opacity=0)
                for node in [self.vis_model.nodes[1][0], *self.vis_model.nodes[0]]
            ]
        )
        self.wait()

        highlight_lines2 = build_layer_lines(
            self.vis_model.input_nodes,
            self.vis_model.nodes[1][1],
            [0.5, -1],
            colors=[colors.GREEN, colors.RED],
            start=RIGHT,
            end=LEFT,
            inflate_opacities=1,
            stuck=False,
            width=8,
        )
        all_neurons.add(highlight_lines2)

        # highlighting input neurons and second output neuron
        self.play(
            *[
                node.animate.set_stroke(colors.ORANGE, 6).set_fill(colors.DESERT, 0.5)
                for node in [self.vis_model.nodes[1][1], *self.vis_model.nodes[0]]
            ],
            FadeIn(highlight_lines2),
        )

        x_text2, y_text2 = get_dot_text()

        self.add(x_text2, y_text2)

        self.play(x_text2.animate.shift(1.5 * UP + RIGHT * 0.8))

        mul_y_x = mul_tex.copy()
        mul_y_x.next_to(x_text2, RIGHT, 0.1)
        self.play(Write(mul_y_x))

        # TODO: review the notation to make sure it's consistent with convention
        w_y_x = (
            MathTex(
                r"w_{y}^{x}",
                color=colors.GREEN,
                font_size=25,
            )
            .next_to(mul_y_x, RIGHT, 0.1)
            .set_stroke(width=1)
        )
        self.play(Transform(highlight_lines2[0], w_y_x))
        self.remove(highlight_lines2[0])
        self.add(w_y_x)

        plus_y = plus_tex.copy()
        plus_y.next_to(w_y_x, RIGHT)
        self.play(Write(plus_y))

        self.play(y_text2.animate.next_to(plus_y, RIGHT))

        mul_y_y = mul_tex.copy()
        mul_y_y.next_to(y_text2, RIGHT, 0.1)
        self.play(Write(mul_y_y))

        w_y_y = (
            MathTex(
                r"w_{y}^{y}",
                color=colors.RED,
                font_size=25,
            )
            .next_to(mul_y_y, RIGHT, 0.1)
            .set_stroke(width=1)
        )
        self.play(Transform(highlight_lines2[1], w_y_y))
        self.remove(highlight_lines2[1])
        self.add(w_y_y)

        # consider putting this inside a function since I've used it 3 times
        non_lin_wrapper2 = VGroup()
        non_lin_wrapper2.add(
            MathTex(r"g(", color=colors.BLACK, font_size=25).next_to(
                x_text2, LEFT, 0.15
            )
        )
        non_lin_wrapper2.add(
            MathTex(r")", color=colors.BLACK, font_size=25).next_to(w_y_y, RIGHT, 0.15)
        )
        non_lin_wrapper2.set_stroke(width=1)
        self.play(Write(non_lin_wrapper2))

        equal_y = equals_tex.copy()
        equal_y.next_to(non_lin_wrapper2, RIGHT)
        self.play(Write(equal_y))

        self.play(output_y.animate.next_to(equal_y, RIGHT))

        # unhighlighting input neurons and second output neuron
        self.play(
            *[
                node.animate.set_stroke(colors.BLACK, 4).set_fill(opacity=0)
                for node in [self.vis_model.nodes[1][1], *self.vis_model.nodes[1]]
            ]
        )
        self.wait()

        x_text1_frame = SurroundingRectangle(x_text1, colors.ORANGE, buff=0.05)
        x_text2_frame = SurroundingRectangle(x_text2, colors.ORANGE, buff=0.05)
        y_text1_frame = SurroundingRectangle(y_text1, colors.DARK_RED, buff=0.05)
        y_text2_frame = SurroundingRectangle(y_text2, colors.DARK_RED, buff=0.05)
        frames = VGroup(x_text1_frame, x_text2_frame, y_text1_frame, y_text2_frame)

        self.play(Write(frames))
        self.wait()

        mobs_to_remove = VGroup(
            mul_x_x, mul_x_y, mul_y_x, mul_y_y, plus_x, plus_y, frames
        )
        self.play(FadeOut(mobs_to_remove))

        self.wait()
        self.play(
            x_text1.animate.next_to(w_x_y, ORIGIN),
            x_text2.animate.next_to(w_x_y, ORIGIN),
            y_text1.animate.next_to(w_y_y, ORIGIN),
            y_text2.animate.next_to(w_y_y, ORIGIN),
            w_x_y.animate.next_to(w_x_x, RIGHT),
            w_y_y.animate.next_to(w_y_x, RIGHT),
        )
        self.remove(x_text2, y_text2)

        weights = VGroup(w_x_x, w_x_y, w_y_x, w_y_y)
        weight_brackets = add_brackets(weights).set_color(colors.BLACK)
        weight_matrix = VGroup(weights, weight_brackets)

        inputs = VGroup(x_text1, y_text1)
        self.play(inputs.animate.next_to(weights, 1.5 * RIGHT))
        input_brackets = add_brackets(inputs).set_color(colors.BLACK)
        input_vector = VGroup(inputs, input_brackets)

        self.play(Write(weight_brackets))
        self.play(Write(input_brackets))

        # consider here just introducing tanh right away and then just denote it as tanh()
        non_lin_brackets = add_brackets(
            VGroup(weight_matrix, input_vector), "curved"
        ).set_color(colors.BLACK)

        non_lin_wrapper3 = VGroup(
            MathTex(r"g", color=colors.BLACK, font_size=25)
            .set_stroke(width=1)
            .next_to(non_lin_brackets, LEFT, 0.15),
            non_lin_brackets,
        )
        self.play(
            Transform(VGroup(non_lin_wrapper1, non_lin_wrapper2), non_lin_wrapper3)
        )
        self.remove(non_lin_wrapper1)
        self.remove(non_lin_wrapper2)
        self.add(non_lin_wrapper3)

        equals_final = equals_tex.copy().next_to(non_lin_wrapper3, RIGHT, 0.25)
        self.play(Transform(VGroup(equal_x, equal_y), equals_final))
        self.remove(equal_x)
        self.remove(equal_y)
        self.add(equals_final)

        outputs = VGroup(output_x, output_y)
        self.play(outputs.animate.next_to(equals_final, 1.5 * RIGHT))
        output_brackets = add_brackets(outputs).set_color(colors.BLACK)
        output_vector = VGroup(outputs, output_brackets)
        self.play(Write(output_brackets))
        self.wait()

        full_operation_text = VGroup(
            output_vector,
            equals_final,
            non_lin_wrapper3,
            weight_matrix,
            input_vector,
        )

        centered_position = [all_neurons.get_x(), full_operation_text.get_y(), 0]
        self.play(full_operation_text.animate.move_to(centered_position))
        self.wait()

        mobs_to_remove = VGroup(
            full_operation_text,
            tracked_dot,
            trans_dot,
            self.vis_model.nodes[:2],
            self.vis_model.lines[0],
            trans_text,
            dot_text,
        )
        self.play(FadeOut(mobs_to_remove))
        self.wait()

    def transform(self):

        code = r"""import torch
from torch import nn

class SpiralsClassifier2D(nn.Module):
    def __init__(self) -> None:
        super(SpiralsClassifier2D, self).__init__()
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

        # putting the ax back to the center of the screen and scaling it up a bit
        self.play(
            VGroup(self.ax, self.blue_spiral, self.red_spiral)
            .animate.move_to(ORIGIN)
            .scale(1.4)
        )
        self.wait()

        model_code.height = self.ax.height * 0.9

        # display the code
        self.play(Write(model_code))

        # arrange the code and the axes to be next to each other
        full_graph = VGroup(self.ax, self.blue_spiral, self.red_spiral)
        self.play(Group(full_graph, model_code).animate.arrange(RIGHT, buff=1))
        self.wait()

        grid = Grid(self.ax, [-3.5, 3.5, 0.5], [-3.5, 3.5, 0.5])

        grid.set_color(colors.BLACK)
        grid.grid_lines.set_stroke(width=0.5)

        self.play(Write(grid.grid_lines))
        self.wait()

        blue_outputs = self.model(torch.tensor(self.blue_spiral_array).float())
        red_outputs = self.model(torch.tensor(self.red_spiral_array).float())
        grid_outputs = self.model(torch.tensor(grid.array).float())

        for blue_output, red_output, grid_output in zip(
            blue_outputs[:-1], red_outputs[:-1], grid_outputs[:-1]
        ):
            new_blue_spiral = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(self.blue_spiral, blue_output.detach())
            ]
            new_red_spiral = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(self.red_spiral, red_output.detach())
            ]
            new_grid = [
                dot.animate.move_to(self.ax.c2p(*pos))
                for dot, pos in zip(grid.submobjects, grid_output.detach())
            ]

            self.play(
                *new_blue_spiral,
                *new_red_spiral,
                *new_grid,
                run_time=3,
                rate_func=rate_functions.linear,
            )

        self.wait()

        wf, bf = list(self.model.parameters())[-2:]
        wf, bf = wf.squeeze().detach(), bf.squeeze().detach()

        # perhaps make it clear that that I'm scaling it down to be visible in screen
        scale_factor = 2e-1
        new_blue_spiral = [
            dot.animate.move_to(self.ax.c2p(pos * scale_factor, 0))
            for dot, pos in zip(self.blue_spiral, blue_outputs[-1].detach())
        ]
        new_red_spiral = [
            dot.animate.move_to(self.ax.c2p(pos * scale_factor, 0))
            for dot, pos in zip(self.red_spiral, red_outputs[-1].detach())
        ]

        new_grid = [
            dot.animate.move_to(self.ax.c2p(pos * scale_factor, 0))
            for dot, pos in zip(grid.submobjects, grid_outputs[-1].detach())
        ]

        self.play(
            *new_blue_spiral,
            *new_red_spiral,
            *new_grid,
            rate_func=rate_functions.linear,
        )
        self.wait()

        new_blue_spiral = [
            dot.animate.move_to(self.ax.c2p(*pos))
            for dot, pos in zip(self.blue_spiral, blue_outputs[-2].detach())
        ]
        new_red_spiral = [
            dot.animate.move_to(self.ax.c2p(*pos))
            for dot, pos in zip(self.red_spiral, red_outputs[-2].detach())
        ]

        new_grid = [
            dot.animate.move_to(self.ax.c2p(*pos))
            for dot, pos in zip(grid.submobjects, grid_outputs[-2].detach())
        ]

        self.play(
            *new_blue_spiral,
            *new_red_spiral,
            *new_grid,
            rate_func=rate_functions.linear,
        )
        self.wait()

        line = Line(
            *[self.ax.c2p(x, -wf[0] / wf[1] * x - bf / wf[1]) for x in [-2.5, 2.5]]
        ).set_stroke(color=colors.BLACK)

        self.play(Create(line))
        self.wait()
        self.play(FadeOut(line))
        self.wait()

        new_blue_spiral = [
            dot.animate.move_to(self.ax.c2p(*pos))
            for dot, pos in zip(self.blue_spiral, self.blue_spiral_array)
        ]
        new_red_spiral = [
            dot.animate.move_to(self.ax.c2p(*pos))
            for dot, pos in zip(self.red_spiral, self.red_spiral_array)
        ]

        new_grid = [
            dot.animate.move_to(self.ax.c2p(*pos))
            for dot, pos in zip(grid.submobjects, grid.array)
        ]

        self.play(
            *new_blue_spiral,
            *new_red_spiral,
            *new_grid,
            rate_func=rate_functions.linear,
        )

        self.wait()
