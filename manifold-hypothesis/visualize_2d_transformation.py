from manim import *
from utils import colors
import numpy as np
import math
from utils.modules import Grid

config.background_color = colors.WHITE


class LinearTransform(Scene):
    def construct(self):
        self.linear_transfom()
        self.wait()

    def linear_transfom(self):
        # build the translation transformation
        ax = Axes(
            x_range=[-6, 6, 2],
            y_range=[-6, 6, 2],
            tips=False,
            x_axis_config={
                "unit_size": 0.5,
                "numbers_to_include": np.arange(-4, 6, 2),
                "font_size": 20,
            },
            y_axis_config={
                "unit_size": 0.5,
                "numbers_to_include": np.arange(-4, 6, 2),
                "font_size": 20,
            },
        ).set_stroke(color=colors.BLACK)

        ax.get_axis(0).numbers.set_color(colors.BLACK).set_stroke(width=1)
        ax.get_axis(1).numbers.set_color(colors.BLACK).set_stroke(width=1)

        ax_border = Rectangle(width=ax.x_length, height=ax.y_length).set_stroke(
            color=colors.BLACK, width=4
        )

        grid = Grid(ax, [-6, 6, 0.5], [-6, 6, 1], lattice_radius=0.04)
        grid.grid_lines.set_stroke(width=0.5)
        labels = ax.get_axis_labels(x_label=Tex("x"), y_label=Tex("y"))

        # build the matrix multiply (mm) text
        mm_text = (
            MathTex(r"Wx", color=colors.WHITE)
            .set_stroke(width=1)
            .move_to(ax.get_corner(UL))
            .shift(RIGHT * 1.8 + DOWN * 0.5)
        )
        mm_text.z_index = 3

        # bounding box around the translation text
        mm_text_box = (
            RoundedRectangle(
                height=mm_text.height * 2 + 0.1,
                width=mm_text.width * 2.8 + 0.3,
                corner_radius=0.2,
            )
            .set_fill(color=colors.BLACK, opacity=0.75)
            .set_stroke(width=0)
            .move_to(mm_text.get_center())
        )
        mm_text_box.z_index = 0

        vector_tracker = [ValueTracker(1), ValueTracker(1)]

        def get_vector():
            return Arrow(
                start=ax.c2p(0, 0),
                end=ax.c2p(*map(lambda m: m.get_value(), vector_tracker)),
                color=colors.PURPLE,
                buff=0,
            )

        vector = always_redraw(get_vector)

        def get_vector_text():
            return Text(
                f"({vector_tracker[0].get_value():.2f}, {vector_tracker[1].get_value():.2f})",
                color=colors.RED,
                font="Fira Code",
                weight=BOLD,
                font_size=15,
            )

        vector_text = always_redraw(get_vector_text)
        vector_text.add_updater(
            lambda x: x.move_to(vector.get_tip().get_center() + UP * 0.25)
        )

        full_graph = VGroup(ax, labels, grid, grid.grid_lines)

        self.play(Write(full_graph), Write(ax_border))
        self.play(Write(vector))
        self.play(Write(vector_text))
        self.play(Write(mm_text_box))
        self.play(Write(mm_text))
        self.wait()
        full_graph.add(vector_text)

        # matrix A performs a sheer transformation
        A = np.array([[1, 1], [0, 1]])
        # matrix B performs a rotation
        B = np.array([[0, -1], [1, 0]])
        # sheer and rotation
        C = A @ B

        # applying C to the x and y trackers
        new_vector = (
            C @ np.array(list(map(lambda m: m.get_value(), vector_tracker)))[:2]
        )
        new_x, new_y = new_vector[0], new_vector[1]

        # applying a matrix multiply to the grid
        grid.array = grid.array @ C.T
        new_grid = [
            dot.animate.move_to(ax.c2p(*pos))
            for dot, pos in zip(grid.submobjects, grid.array)
        ]
        self.play(
            vector_tracker[0].animate.set_value(new_x),
            vector_tracker[1].animate.set_value(new_y),
            *new_grid,
            run_time=3,
        )
        self.wait()


class TanhGraph(Scene):
    def construct(self):
        self.build_tanh_graph()
        self.wait()
        self.show_squishing()
        self.wait()

    def build_tanh_graph(self):
        # build the tanh graph
        ax = Axes(
            x_range=[-6, 6, 2],
            y_range=[-1.5, 1.5, 0.5],
            tips=False,
            y_axis_config={
                "unit_size": 0.5,
                "numbers_to_include": np.arange(-1.0, 1.5, 0.5),
                "font_size": 20,
            },
            x_axis_config={
                "unit_size": 0.5,
                "numbers_to_include": np.arange(-4, 6, 2),
                "font_size": 20,
            },
        ).set_stroke(color=colors.BLACK)

        tanh_text = (
            MathTex(r"\tanh(x)=\frac{e^x - e^{-x}}{e^x + e^{-x}}", color=colors.BLACK)
            .set_stroke(width=1)
            .scale(0.8)
            .move_to(ax.get_corner(UL))
            .shift(DR * 0.7 + RIGHT * 2)
        )

        ax.get_axis(0).numbers.set_color(colors.BLACK).set_stroke(width=1)
        ax.get_axis(1).numbers.set_color(colors.BLACK).set_stroke(width=1)

        self.ax_border = Rectangle(width=ax.x_length, height=ax.y_length).set_stroke(
            color=colors.BLACK, width=4
        )
        graph = ax.plot(math.tanh, color=colors.PURPLE)
        labels = ax.get_axis_labels(x_label=Tex("x"), y_label=Tex("y"))

        t = ValueTracker(0)
        initial_point = [ax.c2p(t.get_value(), math.tanh(t.get_value()))]
        dot = Dot(point=initial_point, radius=0.04, color=colors.BLACK)

        dot.add_updater(
            lambda x: x.move_to(ax.c2p(t.get_value(), math.tanh(t.get_value())))
        )

        def get_dot_text():
            return Text(
                f"{math.tanh(t.get_value()):.2f}",
                color=colors.RED,
                font="Fira Code",
                weight=BOLD,
                font_size=15,
            )

        dot_text = always_redraw(get_dot_text)
        dot_text.add_updater(lambda x: x.next_to(dot, UR))

        full_graph = VGroup(ax, labels, graph, dot, tanh_text)

        # make the dot in front of the graph
        dot.z_index = 3
        graph.z_index = 0

        self.play(Write(VGroup(ax, labels, tanh_text, self.ax_border, dot)))
        self.wait()
        self.play(Write(graph))
        self.wait()
        self.play(Write(dot_text))
        full_graph.add(dot_text)
        self.play(t.animate.set_value(3), run_time=3)
        self.wait()
        self.play(t.animate.set_value(-3), run_time=3)
        self.wait()
        dot_text.clear_updaters()
        self.play(Unwrite(full_graph), Unwrite(dot_text))

    def show_squishing(self):

        fixed_grid = NumberPlane(
            y_length=self.ax_border.height,
            x_length=self.ax_border.width,
            y_range=[-2, 2, 0.5],
            x_range=[-3.5, 3.5, 0.5],
            background_line_style={
                "stroke_width": 1,
                "stroke_opacity": 0.75,
            },
            y_axis_config={
                "unit_size": 0.5,
                "include_numbers": True,
                "numbers_to_include": [-1, 1],
                "font_size": 20,
            },
            x_axis_config={
                "unit_size": 0.5,
                "include_numbers": True,
                "numbers_to_include": [-1, 1],
                "font_size": 20,
            },
            faded_line_ratio=1,
        ).set_stroke(color=colors.GRAY, width=1)

        fixed_grid.axes.set_color(colors.BLACK).set_stroke(width=2)

        grid = Grid(fixed_grid, [-3.5, 3.5, 0.5], [-2, 2, 0.5], lattice_radius=0.04)

        grid.set_color(colors.BLACK)
        grid.grid_lines.set_stroke(width=0.5)

        A = np.array(
            [
                [1, -1],
                [0, 1],
            ]
        )

        self.play(Write(fixed_grid))
        self.wait()
        self.play(Write(VGroup(grid.grid_lines, grid.submobjects)))
        self.wait()

        # applying the linear transformation to the grid
        grid.array = grid.array @ A.T
        new_grid = [
            dot.animate.move_to(fixed_grid.c2p(*pos))
            for dot, pos in zip(grid.submobjects, grid.array)
        ]
        self.play(*new_grid)
        self.wait()

        # applying the tanh function to the grid
        grid.array = np.tanh(grid.array)
        new_grid = [
            dot.animate.move_to(fixed_grid.c2p(*pos))
            for dot, pos in zip(grid.submobjects, grid.array)
        ]
        self.play(*new_grid)
        self.wait()

        grid.submobjects.clear_updaters()
        self.play(Unwrite(grid.submobjects), Unwrite(grid.grid_lines))
        self.play(Unwrite(fixed_grid))
        self.play(Unwrite(self.ax_border))


class ReLUGraph(Scene):
    def construct(self):
        self.build_relu_graph()
        self.wait()

    def build_relu_graph(self):
        # build the relu graph
        ax = Axes(
            x_range=[-6, 6, 2],
            y_range=[-6, 6, 2],
            tips=False,
            x_axis_config={
                "unit_size": 0.5,
                "numbers_to_include": np.arange(-4, 6, 2),
                "font_size": 20,
            },
            y_axis_config={
                "unit_size": 0.5,
                "numbers_to_include": np.arange(-4, 6, 2),
                "font_size": 20,
            },
        ).set_stroke(color=colors.BLACK)

        relu_text = (
            MathTex(r"\text{relu}(x)=\text{max}(0, x)", color=colors.BLACK)
            .set_stroke(width=1)
            .scale(0.8)
            .move_to(ax.get_corner(UL))
            .shift(DR * 0.7 + RIGHT * 1.8)
        )

        ax.get_axis(0).numbers.set_color(colors.BLACK).set_stroke(width=1)
        ax.get_axis(1).numbers.set_color(colors.BLACK).set_stroke(width=1)

        self.ax_border = Rectangle(width=ax.x_length, height=ax.y_length).set_stroke(
            color=colors.BLACK, width=4
        )
        graph = ax.plot(lambda x: max(0, x), color=colors.PURPLE, use_smoothing=False)
        labels = ax.get_axis_labels(x_label=Tex("x"), y_label=Tex("y"))

        t = ValueTracker(0)
        initial_point = [ax.c2p(t.get_value(), max(0, t.get_value()))]
        dot = Dot(point=initial_point, radius=0.04, color=colors.BLACK)
        dot.add_updater(
            lambda x: x.move_to(ax.c2p(t.get_value(), max(0, t.get_value())))
        )

        def get_dot_text():
            return Text(
                f"{max(0, t.get_value()):.2f}",
                color=colors.RED,
                font="Fira Code",
                weight=BOLD,
                font_size=15,
            )

        dot_text = always_redraw(get_dot_text)
        dot_text.add_updater(lambda x: x.next_to(dot, UR))

        full_graph = VGroup(ax, labels, graph, dot, relu_text)

        self.play(Write(full_graph), Write(self.ax_border))
        self.play(Write(dot_text))
        full_graph.add(dot_text)
        self.play(t.animate.set_value(3), run_time=3)
        self.wait()
        self.play(t.animate.set_value(-3), run_time=3)
        self.wait()
        dot_text.clear_updaters()
        self.play(Unwrite(full_graph), Unwrite(dot_text))


class LeakyReLUGraph(Scene):
    def construct(self):
        self.build_leaky_relu_graph()
        self.wait()

    def build_leaky_relu_graph(self):
        # build the leaky relu graph
        ax = Axes(
            x_range=[-6, 6, 2],
            y_range=[-6, 6, 2],
            tips=False,
            x_axis_config={
                "unit_size": 0.5,
                "numbers_to_include": np.arange(-4, 6, 2),
                "font_size": 20,
            },
            y_axis_config={
                "unit_size": 0.5,
                "numbers_to_include": np.arange(-4, 6, 2),
                "font_size": 20,
            },
        ).set_stroke(color=colors.BLACK)

        leaky_relu_text = (
            MathTex(r"\text{leaky\_relu}(x)=\text{max}(ax, x)", color=colors.BLACK)
            .set_stroke(width=1)
            .scale(0.8)
            .move_to(ax.get_corner(UL))
            .shift(DR * 0.7 + RIGHT * 2)
        )

        ax.get_axis(0).numbers.set_color(colors.BLACK).set_stroke(width=1)
        ax.get_axis(1).numbers.set_color(colors.BLACK).set_stroke(width=1)

        self.ax_border = Rectangle(width=ax.x_length, height=ax.y_length).set_stroke(
            color=colors.BLACK, width=4
        )
        graph = ax.plot(
            lambda x: max(0.1 * x, x), color=colors.PURPLE, use_smoothing=False
        )
        labels = ax.get_axis_labels(x_label=Tex("x"), y_label=Tex("y"))

        t = ValueTracker(0)
        initial_point = [ax.c2p(t.get_value(), max(0.1 * t.get_value(), t.get_value()))]
        dot = Dot(point=initial_point, radius=0.04, color=colors.BLACK)
        dot.add_updater(
            lambda x: x.move_to(
                ax.c2p(t.get_value(), max(0.1 * t.get_value(), t.get_value()))
            )
        )

        def get_dot_text():
            return Text(
                f"{max(0.1 * t.get_value(), t.get_value()):.2f}",
                color=colors.RED,
                font="Fira Code",
                weight=BOLD,
                font_size=15,
            )

        dot_text = always_redraw(get_dot_text)
        dot_text.add_updater(lambda x: x.next_to(dot, UR))

        full_graph = VGroup(ax, labels, graph, dot, leaky_relu_text)

        self.play(Write(full_graph), Write(self.ax_border))
        self.play(Write(dot_text))
        full_graph.add(dot_text)
        self.play(t.animate.set_value(3), run_time=3)
        self.wait()
        self.play(t.animate.set_value(-3), run_time=3)
        self.wait()
        dot_text.clear_updaters()
        self.play(Unwrite(full_graph), Unwrite(dot_text))


class AffineNonLinearTransform(Scene):
    def construct(self):
        self.affine_nonlinear_transform()
        self.wait()

    def affine_nonlinear_transform(self):
        # build the translation transformation
        ax = Axes(
            x_range=[-6, 6, 2],
            y_range=[-6, 6, 2],
            tips=False,
            x_axis_config={
                "unit_size": 0.5,
                "numbers_to_include": np.arange(-4, 6, 2),
                "font_size": 20,
            },
            y_axis_config={
                "unit_size": 0.5,
                "numbers_to_include": np.arange(-4, 6, 2),
                "font_size": 20,
            },
        ).set_stroke(color=colors.BLACK)

        ax.get_axis(0).numbers.set_color(colors.BLACK).set_stroke(width=1)
        ax.get_axis(1).numbers.set_color(colors.BLACK).set_stroke(width=1)

        ax_border = Rectangle(width=ax.x_length, height=ax.y_length).set_stroke(
            color=colors.BLACK, width=4
        )

        grid = Grid(ax, [-6, 6, 0.5], [-6, 6, 1], lattice_radius=0.04)
        labels = ax.get_axis_labels(x_label=Tex("x"), y_label=Tex("y"))

        # build the translation text
        translation_text = (
            MathTex(r"x + b", color=colors.WHITE)
            .set_stroke(width=1)
            .move_to(ax.get_corner(UL))
            .shift(RIGHT * 1.8 + DOWN * 0.5)
        )
        translation_text.z_index = 3

        # bounding box around the translation text
        translation_text_box = (
            RoundedRectangle(
                height=translation_text.height * 2 + 0.1,
                width=translation_text.width * 2.8 + 0.3,
                corner_radius=0.2,
            )
            .set_fill(color=colors.BLACK, opacity=0.75)
            .move_to(translation_text.get_center())
            .set_stroke(width=0)
        )
        translation_text_box.z_index = 0

        x_tracker = ValueTracker(0)
        y_tracker = ValueTracker(0)
        initial_point = [ax.c2p(x_tracker.get_value(), y_tracker.get_value())]
        dot = Dot(point=initial_point, radius=0.08, color=colors.PURPLE)
        dot.add_updater(
            lambda x: x.move_to(ax.c2p(x_tracker.get_value(), y_tracker.get_value(), 0))
        )

        def get_dot_text():
            return Text(
                f"({x_tracker.get_value():.2f}, {y_tracker.get_value():.2f})",
                color=colors.RED,
                font="Fira Code",
                weight=BOLD,
                font_size=15,
            )

        dot_text = always_redraw(get_dot_text)
        dot_text.add_updater(lambda x: x.move_to(dot.get_center() + UP * 0.25))

        grid.grid_lines.set_stroke(width=1)
        full_graph = VGroup(ax, labels, grid, grid.grid_lines, dot)

        self.play(Write(full_graph), Write(ax_border))
        self.play(Write(dot_text))
        self.play(Write(translation_text_box))
        self.play(Write(translation_text))
        self.wait()
        full_graph.add(dot_text)

        # applying a translation to the grid
        grid.array = grid.array + [3, 2]
        new_grid = [
            dot.animate.move_to(ax.c2p(*pos))
            for dot, pos in zip(grid.submobjects, grid.array)
        ]
        self.play(
            x_tracker.animate.set_value(3),
            y_tracker.animate.set_value(2),
            *new_grid,
            run_time=3,
        )
        self.wait()

        # shifting the grid back to origin
        grid.array = grid.array + [-3, -2]
        new_grid = [
            dot.animate.move_to(ax.c2p(*pos))
            for dot, pos in zip(grid.submobjects, grid.array)
        ]
        new_x, new_y = x_tracker.get_value() - 3, y_tracker.get_value() - 2
        self.play(
            x_tracker.animate.set_value(new_x),
            y_tracker.animate.set_value(new_y),
            *new_grid,
            run_time=3,
        )
        self.wait()

        affine_text = (
            MathTex(r"Wx + b", color=colors.WHITE)
            .set_stroke(width=1)
            .move_to(ax.get_corner(UL))
            .shift(RIGHT * 1.8 + DOWN * 0.5)
        )

        # matrix A performs a sheer transformation
        A = np.array([[1, 1], [0, 1]])
        # matrix B performs a rotation
        B = np.array([[0, -1], [1, 0]])
        # sheer and rotation
        C = A @ B

        # applying C to the x and y trackers
        new_vector = np.array([x_tracker.get_value(), y_tracker.get_value()]) @ C
        new_x, new_y = new_vector[0], new_vector[1]

        # apply the transformation to the grid
        grid.array = grid.array @ C.T
        new_grid = [
            dot.animate.move_to(ax.c2p(*pos))
            for dot, pos in zip(grid.submobjects, grid.array)
        ]

        self.play(Transform(translation_text, affine_text))
        self.wait()
        self.play(
            x_tracker.animate.set_value(new_x),
            y_tracker.animate.set_value(new_y),
            *new_grid,
            run_time=3,
        )
        self.wait()

        # applying a second translation to the grid
        grid.array = grid.array + [3, 3]
        new_grid = [
            dot.animate.move_to(ax.c2p(*pos))
            for dot, pos in zip(grid.submobjects, grid.array)
        ]
        new_x, new_y = x_tracker.get_value() + 3, y_tracker.get_value() + 3
        self.play(
            x_tracker.animate.set_value(new_x),
            y_tracker.animate.set_value(new_y),
            *new_grid,
            run_time=3,
        )
        self.wait()

        # transform the text to include a tanh function
        tanh_text = (
            MathTex(r"\tanh(Wx + b)", color=colors.WHITE)
            .set_stroke(width=1)
            .move_to(ax.get_corner(UL))
            .shift(RIGHT * 1.8 + DOWN * 0.5)
        )
        self.play(Transform(translation_text, tanh_text))
        self.wait()

        # perform tanh transformation on the grid
        grid.array = np.tanh(grid.array)
        new_grid = [
            dot.animate.move_to(ax.c2p(*pos))
            for dot, pos in zip(grid.submobjects, grid.array)
        ]
        new_x, new_y = np.tanh(x_tracker.get_value()), np.tanh(y_tracker.get_value())
        self.play(
            x_tracker.animate.set_value(new_x),
            y_tracker.animate.set_value(new_y),
            *new_grid,
            run_time=3,
        )
        self.wait()

        dot_text.clear_updaters()
        self.play(FadeOut(full_graph), FadeOut(dot_text))
        self.play(Unwrite(VGroup(ax_border, translation_text_box, translation_text)))
