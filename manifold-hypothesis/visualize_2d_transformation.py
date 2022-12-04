from manim import *
import colors
import numpy as np

config.background_color = colors.WHITE


class VisualizeTransformation(Scene):
    def construct(self):
        self.linear_transfom()
        self.wait()

    def linear_transfom(self):
        grid = NumberPlane(
            y_length=8 * 3,
            y_range=[-8, 8, 0.5],
            x_length=14 * 3,
            x_range=[-14, 14, 0.5],
            background_line_style={
                "stroke_width": 1,
                "stroke_opacity": 0.75,
            },
            faded_line_ratio=1,
        ).set_stroke(color=colors.BLACK)
        fixed_grid = grid.copy().set_stroke(color=colors.DESERT)
        fixed_grid.axes.set_stroke(width=4, color=colors.BLACK)
        grid.axes.set_stroke(width=0)

        dot = Dot(fixed_grid.c2p(1, -1, 0)).set_opacity(0)

        def get_vector():
            vector = (
                Arrow(ORIGIN, dot.get_center(), buff=0)
                .set_stroke(width=4, color=colors.ORANGE)
                .set_fill(color=colors.ORANGE, opacity=1)
            )
            return vector

        vector = always_redraw(get_vector)

        self.play(Write(grid))
        self.add(fixed_grid)
        self.wait()

        self.play(Write(vector))

        A = np.array([[0.25, 0.5], [0.5, 0.25]])

        self.play(
            grid.animate.apply_matrix(A),
            dot.animate.move_to([*(A @ dot.get_center()[0:2]), 0]),
        )


class TanhGraph(Scene):
    def construct(self):
        self.tanh_visualization()
        self.wait()

    def tanh_visualization(self):
        pass
