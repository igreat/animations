from manim import *
from video_utils import *
from PIL import Image
import random
from pytorch_utils.layer import Layer

# consider turning a lot of the code here into
# functions for more readability and maintainability

# REMEMBER THAT YOU CAN DEFINE FUNCTIONS INSIDE THE CLASS YOU DUMBASS!!
# therefore, try to separate the scene into different parts, each defined in
# its own function!
# use self.variable for objects you want to use across scenes

# consider chaning all latex to more bold
# consider changing MSE to just cost function or E
# to make it clear that we can choose any differentiable erro function

# consider changing the fill color of the nodes to the activation of sigmoid
sigma = "σ"


class LogisticRegression(Scene):
    def construct(self):
        self.camera.background_color = colors.WHITE

        bird_pil = Image.open("assets/bird4.png").resize((16, 16))
        bird_image = np.array(bird_pil)

        image_pixels = VGroup()
        # is this row or column?
        for row in bird_image:
            for pixel in row:
                image_pixels.add(
                    Square(
                        color=rgb_to_color(pixel[:3] / 255),
                        fill_opacity=1,
                    )
                )

        image_pixels.arrange_in_grid(cols=16, buff=0)
        image_pixels.height = 6
        self.play(FadeIn(image_pixels))
        self.wait()
        self.play(image_pixels.animate.arrange_in_grid(cols=1, buff=0).shift(LEFT * 4))
        self.wait()

        focus_pixels = VGroup(*image_pixels[:4], *image_pixels[-4:])

        focus_pixels.arrange_in_grid(cols=1, buff=0.1)
        focus_pixels.height = 6
        self.play(Transform(image_pixels, focus_pixels))

        # getting the original array because colors seem to be messed up
        focus_pixels_array = np.concatenate(
            (bird_image.reshape(16 * 16, 3)[:4], bird_image.reshape(16 * 16, 3)[-4:]),
            axis=0,
        )

        red_slice = VGroup()
        green_slice = VGroup()
        blue_slice = VGroup()

        for pixel in focus_pixels_array:
            red, green, blue = pixel / 255
            red_slice.add(Square(color=RED, fill_opacity=red))
            green_slice.add(Square(color=GREEN, fill_opacity=green))
            blue_slice.add(Square(color=BLUE, fill_opacity=blue))

        red_slice.arrange_in_grid(cols=1, buff=0.1).set_stroke(width=0).move_to(
            focus_pixels.get_center()
        )
        green_slice.arrange_in_grid(cols=1, buff=0.1).set_stroke(width=0).move_to(
            focus_pixels.get_center()
        )
        blue_slice.arrange_in_grid(cols=1, buff=0.1).set_stroke(width=0).move_to(
            focus_pixels.get_center()
        )

        red_slice.height = 6
        green_slice.height = 6
        blue_slice.height = 6

        self.play(
            FadeIn(red_slice),
            red_slice.animate.shift(LEFT * 0.75),
            FadeIn(green_slice),
            FadeIn(blue_slice),
            blue_slice.animate.shift(RIGHT * 0.75),
            FadeOut(image_pixels),
        )

        rgb_flattened = VGroup(red_slice, green_slice, blue_slice)

        self.play(rgb_flattened.animate.arrange_in_grid(cols=1, buff=0))

        self.play(
            FadeOut(red_slice),
            red_slice.animate.shift(UP),
            FadeOut(blue_slice),
            blue_slice.animate.shift(DOWN),
            green_slice.animate.set_color(color=colors.BLACK),
        )
        self.demo_pixels = green_slice

        self.play(self.demo_pixels.animate.move_to([-5, 0, 0]))

        self.node = Circle(radius=0.5, color=colors.GRAY, fill_opacity=1).set_stroke(
            color=colors.BLACK, width=4
        )
        self.node.z_index = 1
        random.seed(4)
        weights = [(random.random() - 0.5) * 2 for _ in range(len(self.demo_pixels))]
        self.connection_lines = build_layer_lines(
            self.demo_pixels,
            VGroup(self.node),
            colors=[colors.GREEN, colors.RED],
            start=RIGHT,
            end=LEFT,
            opacities=weights,
            always_back=True,
        )

        pixel_values = VGroup()
        self.pixel_nums = []
        for pixel in self.demo_pixels:
            pixel_values.add(
                Text(
                    f"{pixel.fill_opacity:.2f}",
                    font="Fira Code",
                    weight=BOLD,
                    font_size=15,
                ).move_to(pixel.get_center())
            )
            self.pixel_nums.append(pixel.fill_opacity)

        self.play(Write(self.node), Write(self.connection_lines))
        self.play(Write(pixel_values))
        self.wait()

        self.play(
            pixel_values.animate.move_to([4, 0.5, 0])
            .scale(2)
            .set_color(colors.BLACK)
            .arrange_in_grid(cols=1, buff=0.5)
        )
        value_multiply_text = VGroup()

        # consider color coating this text
        # for example * might have different color and so on
        for pixel, weight in zip(pixel_values, weights):
            weight_text = f"{weight:.2f}"
            if weight >= 0:
                weight_text = f" {weight:.2f}"
            new_text = Text(
                f"{pixel.text} * {weight_text}",
                color=colors.BLACK,
                t2c={
                    "[:4]": colors.BLACK,
                    "[4:6]": colors.RED,
                    "[6:]": colors.ORANGE,
                },
                font="Fira Code",
                weight=BOLD,
                font_size=15,
            )
            value_multiply_text.add(new_text)

        value_multiply_text.scale(2).arrange_in_grid(cols=1, buff=0.5).move_to(
            pixel_values.get_center()
        )
        plus = Text(
            "+",
            color=colors.BLACK,
            font="Fira Code",
            weight=BOLD,
            font_size=15,
        )
        pluses = VGroup(*[plus.copy() for _ in range(len(weights) - 1)])
        pluses.scale(2).arrange_in_grid(cols=1, buff=0.55).move_to(
            value_multiply_text.get_left()
        ).shift(LEFT * 0.5)

        self.play(Transform(pixel_values, value_multiply_text))

        self.play(Write(pluses))

        equals_line = (
            Line(*[value_multiply_text.get_corner(d) for d in (DL, DR)])
            .set_stroke(color=colors.BLACK, width=4)
            .shift(DOWN * 0.5)
        )

        self.play(Write(equals_line))

        actual_sum = sum(
            [float(pixel.text) * weight for pixel, weight in zip(pixel_values, weights)]
        )
        self.sum_text = (
            Text(
                f"{actual_sum:.2f}",
                color=colors.BLACK,
                font="Fira Code",
                weight=BOLD,
                font_size=15,
            )
            .scale(2)
            .move_to(equals_line.get_center())
            .shift(DOWN * 0.5)
        )

        self.play(Write(self.sum_text))
        self.wait()
        self.play(
            FadeOut(pixel_values, equals_line, pluses, shift=UP),
        )
        self.play(self.sum_text.animate.next_to(self.node, UP))

        sigm = MathTex(r"\sigma(x) = \frac{1}{1 + e^{-x}}", color=colors.BLACK).next_to(
            self.node, DOWN
        )
        x = self.sum_text.text
        result = f"{sigmoid(float(x)):.2f}"
        sigm_inputted = (
            MathTex(
                r"\sigma(" + x + r") &= \frac{1}{1 + e^{-(" + x + r")}}\\ &= " + result,
                color=colors.BLACK,
            )
            .next_to(sigm, DOWN)
            .scale(0.8)
        )
        self.play(Write(sigm))
        self.wait()
        self.play(Write(sigm_inputted))
        self.activation = VGroup(self.node, self.sum_text, sigm, sigm_inputted)

        self.sigmoid_graph()
        self.wait()
        self.python_bird_frog()
        self.wait(2)

        self.play(*[FadeOut(mob, shift=UP * 5) for mob in self.mobjects])
        self.wait(2)
        self.training_part()
        self.wait(2)
        self.play(*[FadeOut(mob, shift=DOWN * 5) for mob in self.mobjects])
        self.wait(2)
        self.tweaking_weights()
        self.wait(2)

    def sigmoid_graph(self):
        # build the sigmoid graph
        ax = Axes(
            x_range=[-6, 6, 2],
            y_range=[-0.1, 1.1, 0.5],
            x_length=3,
            y_length=2,
            tips=False,
            y_axis_config={
                "unit_size": 0.5,
                "numbers_to_include": [0.5, 1],
                "font_size": 20,
            },
        ).set_stroke(color=colors.BLACK)

        ax.get_axis(1).numbers.set_color(colors.BLACK)
        ax_border = Rectangle(width=ax.x_length, height=ax.y_length).set_stroke(
            color=colors.BLACK, width=4
        )
        graph = ax.plot(sigmoid, color=colors.PURPLE)
        labels = ax.get_axis_labels(x_label=Tex("x"), y_label=Tex("y"))

        t = ValueTracker(0)
        initial_point = [ax.c2p(t.get_value(), sigmoid(t.get_value()))]
        dot = Dot(point=initial_point, radius=0.04, color=colors.BLACK)
        dot.add_updater(
            lambda x: x.move_to(ax.c2p(t.get_value(), sigmoid(t.get_value())))
        )

        def get_dot_text():
            return Text(
                f"{sigmoid(t.get_value()):.2f}",
                color=colors.BLACK,
                font="Fira Code",
                weight=BOLD,
                font_size=15,
            )

        dot_text = always_redraw(get_dot_text)
        dot_text.add_updater(lambda x: x.next_to(dot, UR))

        self.full_graph = (
            VGroup(ax, ax_border, labels, graph, dot).move_to([4, 0, 0]).scale(1.5)
        )

        self.play(
            Write(self.full_graph),
            self.activation.animate.move_to([-1.5, 0, 0]),
            self.demo_pixels.animate.shift(LEFT * 0.5),
        )
        self.play(Write(dot_text))
        self.full_graph.add(dot_text)
        self.play(t.animate.set_value(float(self.sum_text.text)))

    def python_bird_frog(self):
        # make sure you report the issue where >= doesn't work for color coating
        # or maybe even fix it yourself!

        code = """if sigmoid(x) > 0.5:
    result = "bird"
else:
    result = "frog"
"""
        self.bird_frog_code = (
            Code(
                code=code,
                tab_width=2,
                background="rectangle",
                language="Python",
                font="Fira Code",
                style="vs",
                background_stroke_color=colors.BLACK,
                background_stroke_width=4,
            )
            .next_to(self.full_graph, DOWN)
            .shift(UP, LEFT * 0.27)
        )
        self.play(
            FadeIn(self.bird_frog_code, shift=DOWN),
            self.full_graph.animate.shift(UP * 2),
        )

    def training_part(self):
        # consider adding code snippet here

        mse_tex = MathTex(
            r"\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_{i} - \hat{y_{i}})^{2}",
            color=colors.BLACK,
        )
        mse_tex_supplement = MathTex(
            r"y_{i} \\ \hat{y_{i}}",
            color=colors.BLACK,
        ).next_to(mse_tex, DOWN)
        mse_text_supplements = Text(
            ": predicted value\n\n: actual value",
            t2c={
                "[0:1]": colors.BLACK,
                "[1:19]": colors.PURPLE,
                "[19:20]": colors.BLACK,
                "[20:]": colors.PURPLE,
            },
            font_size=25,
            weight=BOLD,
            font="Fira Code",
        )
        mse_supplement = (
            VGroup(mse_tex_supplement, mse_text_supplements)
            .arrange(RIGHT)
            .next_to(mse_tex, DOWN)
            .shift(UP * 0.5)
        )
        self.play(Write(mse_tex))
        self.play(
            Write(mse_supplement),
            mse_tex.animate.shift(UP),
        )
        mse_all = VGroup(mse_tex, mse_supplement)

        self.play(mse_all.animate.move_to([0, 1.5, 0]))

        argmin_tex = MathTex(
            r"\text{argmin}(\text{MSE}(w_{0}, w_{1}, ..., w_{n}))",
            color=colors.DARK_RED,
        ).move_to([0, -3, 0])

        arrow = Arrow(
            start=mse_all.get_critical_point(DOWN) + DOWN * 0.5,
            end=argmin_tex.get_critical_point(UP) + UP * 0.5,
            color=colors.GRAY,
        ).set_stroke(color=colors.BLACK, width=4)
        self.play(Write(arrow))
        self.play(Write(argmin_tex))

    def tweaking_weights(self):
        self.node.move_to(ORIGIN + LEFT * 0.5)

        weights = Group(*[ValueTracker((random.random() - 0.5) * 2) for _ in range(8)])

        # essentailly using a dot as a vector value tracker
        sigm_position = Dot(radius=0, point=self.node.get_bottom() + DOWN * 0.5)

        def weighed_sum(inputs, weights):
            result = 0
            for x, w in zip(inputs, weights):
                result += x * w.get_value()
            return result

        def get_sigm_updated():
            value = weighed_sum(self.pixel_nums, weights)
            result = f"{sigmoid(value):.2f}"
            x = f"{value:.2f}"

            sigm_updated = Text(
                f"{sigma}({x}) = {result}",
                color=colors.BLACK,
                font="Fira Code",
                font_size=30,
                weight=BOLD,
            ).move_to(sigm_position.get_center())
            return sigm_updated

        error_text_position = Dot(radius=0)

        def get_error_text():
            value = weighed_sum(self.pixel_nums, weights)
            output = sigmoid(value)

            error = (1 - output) ** 2
            error_text = Text(
                f"mse_error({output:.2f}) = {error:.2f}",
                color=colors.BLACK,
                font="Fira Code",
                font_size=30,
                weight=BOLD,
            ).move_to(error_text_position.get_center())

            return error_text

        error_text = always_redraw(get_error_text)

        self.connection_lines = build_layer_lines(
            self.demo_pixels,
            VGroup(self.node),
            colors=[colors.GREEN, colors.RED],
            opacities=weights,
            start=RIGHT,
            end=LEFT,
        )
        self.connection_lines.shift(RIGHT * 2)
        self.node.shift(RIGHT * 2)
        self.demo_pixels.shift(RIGHT * 2)

        self.play(
            FadeIn(self.connection_lines, self.node, self.demo_pixels, shift=RIGHT * 2)
        )
        self.play(
            self.node.animate.shift(RIGHT * 2), self.demo_pixels.animate.shift(LEFT)
        )

        sigm_updated = always_redraw(get_sigm_updated)
        sigm_position.shift(UP)
        sigm_updated.resume_updating()

        self.play(Write(sigm_updated))

        error_text_position.next_to(sigm_position, DOWN * 4)

        error_text.resume_updating()
        self.play(Write(error_text))

        clarity_box = (
            RoundedRectangle(width=6, height=2)
            .set_fill(colors.BLACK, opacity=0.1)
            .set_stroke(width=0)
            .move_to(
                (sigm_position.get_center() + error_text_position.get_center()) / 2
            )
        )
        self.play(FadeIn(clarity_box))
        for _ in range(10):
            animations = []
            for weight in weights:
                animations.append(weight.animate.set_value((random.random() - 0.5) * 2))

            self.play(*animations)


class FullyConnectedNN(Scene):
    def construct(self):
        self.camera.background_color = colors.WHITE

        self.generalize_scene()
        self.wait(2)

    def generalize_scene(self):
        input_length = 32
        input_nodes = VGroup(
            *[Circle(radius=0.25, color=colors.BLACK) for _ in range(8)]
        )
        input_nodes.arrange(UP).move_to([-6, 0, 0])

        hidden_layer1_init = Layer(input_length, 1, self, input_displayed=8)
        hidden_layer1 = Layer(input_length, 4, self, input_displayed=8)
        hidden_layer2 = Layer(hidden_layer1.layer.in_features, 6, self)
        output_layer_init = Layer(hidden_layer2.layer.in_features, 1, self)
        output_layer = Layer(hidden_layer2.layer.in_features, 3, self)

        # hidden_layer1.nodes.move_to([-2, 0, 0])
        hidden_layer2.nodes.move_to([2, 0, 0])
        output_layer_init.nodes.move_to([6, 0, 0])
        output_layer.nodes.move_to([6, 0, 0])

        hidden_layer1.build_lines(input_nodes)
        hidden_layer2.build_lines(hidden_layer1.nodes)
        output_layer.build_lines(hidden_layer2.nodes)

        hidden_layer1.lines.resume_updating()
        hidden_layer2.lines.resume_updating()
        output_layer.lines.resume_updating()

        self.play(FadeIn(input_nodes, shift=UP))

        self.play(FadeIn(hidden_layer1_init.nodes, shift=UP))
        self.wait()

        self.play(Transform(hidden_layer1_init.nodes, hidden_layer1.nodes))
        self.play(Create(hidden_layer1.lines))
        self.remove(hidden_layer1_init.nodes)
        self.add(hidden_layer1.nodes)
        self.play(hidden_layer1.nodes.animate.move_to([-2, 0, 0]))
        hidden_layer2.lines.resume_updating()

        self.play(FadeIn(hidden_layer2.nodes, shift=UP))
        self.play(Create(hidden_layer2.lines))

        self.play(FadeIn(output_layer_init.nodes, shift=UP))

        self.play(Transform(output_layer_init.nodes, output_layer.nodes))
        self.play(Create(output_layer.lines))
        self.remove(output_layer_init.nodes)
        self.add(output_layer.nodes)
