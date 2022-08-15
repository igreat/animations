from manim import *
from video_utils import *
from PIL import Image
import torch
from pytorch_utils.layer import Layer, build_labels

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

        self.linear = Layer(8, 1, self)
        self.linear.layer.float()
        self.linear.layer.requires_grad_(False)
        self.linear.build_lines(self.demo_pixels, colors=[colors.GREEN, colors.RED])
        self.linear.lines.resume_updating()
        self.linear.nodes.scale(2)

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

        self.play(Write(pixel_values))
        self.play(Write(self.linear.nodes))
        self.linear.lines.resume_updating()
        self.play(Write(self.linear.lines))
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
        for pixel, weight in zip(pixel_values, self.linear.weights):
            weight_text = f"{weight.get_value():.2f}"
            if weight.get_value() >= 0:
                weight_text = f" {weight.get_value():.2f}"
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
        pluses = VGroup(*[plus.copy() for _ in range(len(self.linear.weights) - 1)])
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
            [
                float(pixel.text) * weight.get_value()
                for pixel, weight in zip(pixel_values, self.linear.weights)
            ]
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
        self.play(self.sum_text.animate.next_to(self.linear.nodes, UP))

        sigm = (
            MathTex(r"\sigma(x) = \frac{1}{1 + e^{-x}}", color=colors.BLACK)
            .next_to(self.linear.nodes, DOWN)
            .set_stroke(width=1)
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
            .set_stroke(width=1)
        )
        self.play(Write(sigm))
        self.wait()
        self.play(Write(sigm_inputted))
        self.activation = VGroup(self.linear.nodes, self.sum_text, sigm, sigm_inputted)

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

        ax.get_axis(1).numbers.set_color(colors.BLACK).set_stroke(width=1)
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
                line_spacing=1,
                font_size=18,
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
        ).set_stroke(width=1)

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
        ).set_stroke(width=1)

        self.play(Write(mse_tex))
        self.play(
            Write(mse_supplement),
            mse_tex.animate.shift(UP),
        )
        mse_all = VGroup(mse_tex, mse_supplement)

        self.play(mse_all.animate.move_to([0, 1.5, 0]))

        argmin_tex = (
            MathTex(
                r"\text{argmin}(\text{MSE}(w_{0}, w_{1}, ..., w_{n}))",
                color=colors.DARK_RED,
            )
            .set_stroke(width=1)
            .move_to([0, -3, 0])
        )

        arrow = Arrow(
            start=mse_all.get_critical_point(DOWN) + DOWN * 0.5,
            end=argmin_tex.get_critical_point(UP) + UP * 0.5,
            color=colors.GRAY,
        ).set_stroke(color=colors.BLACK, width=4)
        self.play(Write(arrow))
        self.play(Write(argmin_tex))

    def tweaking_weights(self):
        self.linear.nodes.move_to(ORIGIN + LEFT * 0.5)

        # weights = Group(*[ValueTracker((random.random() - 0.5) * 2) for _ in range(8)])

        # essentailly using a dot as a vector value tracker
        sigm_position = Dot(radius=0)

        def get_sigm_updated():
            input = torch.tensor(self.pixel_nums).unsqueeze(0).float()
            value = self.linear.layer(input).item()
            result = f"{sigmoid(value):.2f}"
            x = f"{value:.2f}"

            sigm_updated = Text(
                f"{sigma}({x}) = {result}",
                color=colors.WHITE,
                font="Fira Code",
                font_size=30,
                weight=BOLD,
            ).move_to(sigm_position.get_center())
            return sigm_updated

        error_text_position = Dot(radius=0)

        def get_error_text():
            input = torch.tensor(self.pixel_nums).unsqueeze(0).float()
            value = self.linear.layer(input).item()

            output = sigmoid(value)

            error = (1 - output) ** 2
            error_text = Text(
                f"mse_error({output:.2f}) = {error:.2f}",
                color=colors.WHITE,
                font="Fira Code",
                font_size=30,
                weight=BOLD,
            ).move_to(error_text_position.get_center())

            return error_text

        error_text = always_redraw(get_error_text)

        self.linear.nodes.shift(RIGHT * 3.1)
        self.demo_pixels.shift(RIGHT * 2)
        self.linear.lines.resume_updating()

        self.play(
            FadeIn(
                self.linear.lines,
                self.linear.nodes,
                self.demo_pixels,
                shift=RIGHT * 2,
            )
        )
        self.play(
            self.linear.nodes.animate.shift(RIGHT * 2),
            self.demo_pixels.animate.shift(LEFT),
        )

        sigm_updated = always_redraw(get_sigm_updated)
        sigm_updated.resume_updating()

        clarity_box = (
            RoundedRectangle(width=7, height=2.5)
            .set_fill(colors.BLACK, opacity=0.8)
            .set_stroke(width=0)
        )

        self.play(FadeIn(clarity_box))
        self.play(Write(sigm_updated))

        position_manager = VGroup(sigm_position, error_text_position)
        self.play(position_manager.animate.arrange(DOWN, buff=0.75))

        error_text.resume_updating()
        self.play(Write(error_text))

        self.linear.layer.requires_grad_(True)
        target = torch.tensor([1]).unsqueeze(0).float()
        for _ in range(20):
            self.linear.layer.zero_grad()

            x = torch.tensor(self.pixel_nums).unsqueeze(0).float()
            x = self.linear.layer(x)
            output = torch.sigmoid(x)
            loss = torch.nn.functional.mse_loss(output, target)
            print(loss.item())
            loss.backward()

            with torch.no_grad():
                self.linear.layer.weight -= self.linear.layer.weight.grad
                self.linear.layer.bias -= self.linear.layer.bias.grad

            self.wait(0.5)


# this is the scene where I show how we can generalize from
# logistic regression
class FullyConnectedNN(Scene):
    def construct(self):
        self.camera.background_color = colors.WHITE

        self.build_nn()
        self.wait()
        self.introducing_vectorized_implementation()
        self.wait()
        self.affine_non_linear_demo()
        self.wait(2)
        self.backpropagation()
        self.wait(2)

    def build_nn(self):
        input_length = 32
        self.input_nodes = VGroup(
            *[Circle(radius=0.25, color=colors.BLACK) for _ in range(8)]
        )
        self.input_nodes.arrange(DOWN).move_to([-6, 0, 0])

        hidden_layer1_init = Layer(input_length, 1, self, input_displayed=8)
        self.hidden_layer1 = Layer(input_length, 4, self, input_displayed=8)
        self.hidden_layer2 = Layer(self.hidden_layer1.layer.in_features, 6, self)
        output_layer_init = Layer(self.hidden_layer2.layer.in_features, 1, self)
        self.output_layer = Layer(self.hidden_layer2.layer.in_features, 3, self)

        # hidden_layer1.nodes.move_to([-2, 0, 0])
        self.hidden_layer2.nodes.move_to([2, 0, 0])
        output_layer_init.nodes.move_to([6, 0, 0])
        self.output_layer.nodes.move_to([6, 0, 0])

        self.hidden_layer1.build_lines(self.input_nodes)
        self.hidden_layer2.build_lines(self.hidden_layer1.nodes)
        self.output_layer.build_lines(self.hidden_layer2.nodes)

        self.hidden_layer1.lines.resume_updating()
        self.hidden_layer2.lines.resume_updating()
        self.output_layer.lines.resume_updating()

        self.play(FadeIn(self.input_nodes, shift=UP))

        self.play(FadeIn(hidden_layer1_init.nodes, shift=UP))
        self.wait()

        self.play(Transform(hidden_layer1_init.nodes, self.hidden_layer1.nodes))
        self.play(Create(self.hidden_layer1.lines))
        self.remove(hidden_layer1_init.nodes)
        self.add(self.hidden_layer1.nodes)
        self.play(self.hidden_layer1.nodes.animate.move_to([-2, 0, 0]))
        self.hidden_layer2.lines.resume_updating()

        self.play(FadeIn(self.hidden_layer2.nodes, shift=UP))
        self.play(Create(self.hidden_layer2.lines))

        self.play(FadeIn(output_layer_init.nodes, shift=UP))

        self.play(Transform(output_layer_init.nodes, self.output_layer.nodes))
        self.play(Create(self.output_layer.lines))
        self.remove(output_layer_init.nodes)
        self.add(self.output_layer.nodes)
        self.wait()

        # here is where I will start introducing some notation
        self.input_layer_labels = build_labels(self.input_nodes, 0)

        self.hidden_layer1.build_labels(1)
        self.hidden_layer2.build_labels(2)
        self.output_layer.build_labels(3)

        self.play(Write(self.input_layer_labels), run_time=0.5)
        self.play(Write(self.hidden_layer1.labels), run_time=0.5)
        self.play(Write(self.hidden_layer2.labels), run_time=0.5)
        self.play(Write(self.output_layer.labels), run_time=0.5)

    def introducing_vectorized_implementation(self):

        self.mobs_to_remove = VGroup(
            self.hidden_layer2.nodes,
            self.hidden_layer2.lines,
            self.hidden_layer2.labels,
            self.output_layer.nodes,
            self.output_layer.lines,
            self.output_layer.labels,
        )
        self.play(FadeOut(self.mobs_to_remove))
        self.play(
            self.input_nodes.animate.move_to([-4.5, 0, 0]),
            self.hidden_layer1.nodes.animate.move_to([4.5, 0, 0]),
        )

        rect_width = (
            abs(
                self.input_nodes.get_center()[0]
                - self.hidden_layer1.nodes.get_center()[0]
            )
            - 1
        )
        text_box = (
            RoundedRectangle(
                width=rect_width,
                height=self.input_nodes.height - 3,
            )
            .set_fill(color=colors.BLACK, opacity=0.8)
            .set_stroke(width=0)
        )

        self.play(FadeIn(text_box))

        matrix_operation = MathTex(
            r"""
            z_{0}^{1} = w_{0,0}a_{0}^{0} + w_{0,1}a_{1}^{0} + ... + w_{0,6}a_{6}^{0} + w_{0,7}a_{7}^{0} + b_{0} \\
            z_{1}^{1} =  w_{1,0}a_{0}^{0} + w_{1,1}a_{1}^{0} + ... + w_{1,6}a_{6}^{0} + w_{1,7}a_{7}^{0} + b_{1} \\
            z_{2}^{1} =  w_{2,0}a_{0}^{0} + w_{2,1}a_{1}^{0} + ... + w_{2,6}a_{6}^{0} + w_{2,7}a_{7}^{0} + b_{2} \\
            z_{3}^{1} =  w_{3,0}a_{0}^{0} + w_{3,1}a_{1}^{0} + ... + w_{3,6}a_{6}^{0} + w_{3,7}a_{7}^{0} + b_{3}
            """,
            color=colors.WHITE,
            font_size=30,
        ).set_stroke(width=1)
        z_vec = MathTex(
            r"""
            \begin{bmatrix} 
                z_{0}^{1} \\
                z_{1}^{1} \\
                z_{2}^{1} \\
                z_{3}^{1}
            \end{bmatrix}
        """,
            color=colors.WHITE,
            font_size=30,
        )
        weight_matrix = MathTex(
            r"""
            =
            \begin{bmatrix} 
                w_{0,0} & w_{0,1} & ... & w_{0,6} & w_{0,7} \\
                w_{1,0} & w_{1,1} & ... & w_{1,6} & w_{1,7} \\
                w_{2,0} & w_{2,1} & ... & w_{2,6} & w_{2,7} \\
                w_{3,0} & w_{3,1} & ... & w_{3,6} & w_{3,7}
            \end{bmatrix}
            """,
            color=colors.WHITE,
            font_size=30,
        )
        a_vec = MathTex(
            r"""
            \begin{bmatrix} 
                a_{0}^{0} \\
                a_{1}^{0} \\
                \vdots \\
                a_{6}^{0} \\
                a_{7}^{0}
            \end{bmatrix}
            """,
            color=colors.WHITE,
            font_size=30,
        )
        b_vec = MathTex(
            r"""
            +
            \begin{bmatrix} 
                b_{0} \\
                b_{1} \\
                b_{2} \\
                b_{3}
            \end{bmatrix}
            """,
            color=colors.WHITE,
            font_size=30,
        )
        vectorized_operation = (
            VGroup(z_vec, weight_matrix, a_vec, b_vec)
            .set_stroke(width=1)
            .arrange(RIGHT)
        )
        self.play(Write(matrix_operation))
        self.wait()

        self.play(FadeTransform(matrix_operation, vectorized_operation), run_time=1.5)
        self.remove(matrix_operation)
        self.add(vectorized_operation)

        self.wait()

        a_new_vec = MathTex(
            r"""
            \begin{bmatrix} 
                a_{0}^{1} \\
                a_{1}^{1} \\
                a_{2}^{1} \\
                a_{3}^{1}
            \end{bmatrix}
            =
            """,
            color=colors.WHITE,
            font_size=30,
        ).set_stroke(width=1)
        sigmoid_z = MathTex(
            r"""
            \sigma\left(
            \begin{bmatrix} 
                z_{0}^{1} \\
                z_{1}^{1} \\
                z_{2}^{1} \\
                z_{3}^{1}
            \end{bmatrix}
            \right)
            """,
            color=colors.WHITE,
            font_size=30,
        ).set_stroke(width=1)

        self.play(FadeOut(VGroup(weight_matrix, a_vec, b_vec), shift=2 * RIGHT))
        self.play(z_vec.animate.move_to(ORIGIN))
        self.wait()
        self.play(Transform(z_vec, sigmoid_z))
        self.remove(z_vec)
        self.add(sigmoid_z)

        a_equals_sig = VGroup(a_new_vec, sigmoid_z)

        self.play(FadeIn(a_new_vec), a_equals_sig.animate.arrange(RIGHT))
        self.wait()
        self.play(FadeOut(a_equals_sig))

        functional_form = MathTex(
            r"a^{1} = \sigma\left(Wa^{0} + b\right)", font_size=40, color=colors.WHITE
        ).set_stroke(width=1)

        self.play(Write(functional_form))
        self.wait(2)

        self.play(FadeOut(VGroup(text_box, functional_form), shift=UP))

    def affine_non_linear_demo(self):
        self.play(
            self.input_nodes.animate.move_to([-6, 0, 0]),
            self.hidden_layer1.nodes.animate.move_to([-2, 0, 0]),
        )
        self.play(FadeIn(self.mobs_to_remove, shift=LEFT))

        weights1 = MathTex(
            r"W_{1}",
            font_size=40,
            color=colors.BLACK,
        ).set_stroke(width=1.5)

        weights2 = MathTex(
            r"W_{2}",
            font_size=40,
            color=colors.BLACK,
        ).set_stroke(width=1.5)

        weights3 = MathTex(
            r"W_{3}",
            font_size=40,
            color=colors.BLACK,
        ).set_stroke(width=1.5)

        weights_all = VGroup(weights1, weights2, weights3).arrange(RIGHT, buff=3.4)

        self.play(FadeIn(weights_all, shift=UP))

        affine_transform = (
            Tex(
                r"affine\_transform1",
                font_size=40,
                color=colors.BLACK,
            )
            .set_stroke(width=1.5)
            .move_to([-3.9, -3, 0])
        )

        plus = (
            MathTex(
                r"\circ",
                font_size=40,
                color=colors.BLACK,
            )
            .set_stroke(width=1.5)
            .next_to(affine_transform, RIGHT)
        )

        affine_transform2 = (
            Tex(
                r"affine\_transform2",
                font_size=40,
                color=colors.BLACK,
            )
            .set_stroke(width=1.5)
            .next_to(plus, RIGHT)
        )

        arrow = (
            MathTex(
                r"\rightarrow",
                font_size=40,
                color=colors.BLACK,
            )
            .set_stroke(width=1.5)
            .next_to(affine_transform2, RIGHT)
        )

        affine_transform3 = (
            Tex(
                r"affine\_transform3",
                font_size=40,
                color=colors.BLACK,
            )
            .set_stroke(width=1.5)
            .next_to(arrow, RIGHT)
        )

        non_linear_transform = (
            Tex(
                r"non\_linear\_transform",
                font_size=40,
                color=colors.BLACK,
            )
            .set_stroke(width=1.5)
            .next_to(plus, RIGHT)
        )
        self.whole_network = VGroup(
            self.input_nodes,
            self.input_layer_labels,
            self.hidden_layer1.get_layer_mobs(),
            self.hidden_layer2.get_layer_mobs(),
            self.output_layer.get_layer_mobs(),
            weights_all,
        )

        self.play(
            FadeIn(affine_transform, shift=UP),
            self.whole_network.animate.shift(UP * 0.8),
        )

        self.play(FadeIn(plus, shift=UP))
        self.play(FadeIn(affine_transform2, shift=UP))
        self.play(FadeIn(arrow, shift=RIGHT))
        self.play(FadeIn(affine_transform3, shift=UP))

        # here is where the whole screen will fade away, to
        # show the non-linear-transform circle example
        affine_transform_full = VGroup(
            affine_transform, affine_transform2, affine_transform3, plus, arrow
        )
        self.wait()
        self.play(
            FadeOut(self.whole_network, shift=UP),
            FadeOut(affine_transform_full, shift=DOWN),
        )

        self.wait()
        self.whole_network.move_to(ORIGIN)
        self.play(FadeIn(self.whole_network, shift=DOWN))

        # finish up the part where you need to show non_linear_transform text

    def backpropagation(self):
        back_arrows = VGroup()
        for layer in [self.output_layer, self.hidden_layer2, self.hidden_layer1]:
            back_arrows_layer = VGroup()
            for line in layer.lines:
                unit_vector = -line.get_unit_vector()
                new_arrow = (
                    Arrow(
                        color=colors.BLACK,
                        start=line.get_end() - unit_vector * 0.25,
                        end=line.get_end() + unit_vector * 0.8,
                        max_tip_length_to_length_ratio=0.1,
                    )
                    .set_stroke(width=2, opacity=0.5)
                    .set_fill(opacity=0.5)
                )

                back_arrows_layer.add(new_arrow)

            back_arrows.add(back_arrows_layer)

        for arrows in back_arrows:
            self.play(Write(arrows))

        self.whole_network.add(back_arrows)

        backprop_text = (
            Tex(r"The Backpropagation Algorithm", font_size=40, color=colors.BLACK)
            .set_stroke(width=1)
            .next_to(self.whole_network, UP)
        )

        self.play(FadeIn(backprop_text, shift=UP))
