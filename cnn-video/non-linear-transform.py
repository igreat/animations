from turtle import circle
from manim import *
import colors
import random
import math
import numpy as np


class NonLinearTransform(Scene):
    def construct(self):
        self.camera.background_color = colors.WHITE
        self.original_space()
        self.wait()
        self.feature_transform()
        self.wait()
        self.feature_space()
        self.wait(2)

    def original_space(self):
        ax = Axes(
            x_range=[-2.5, 2.5, 0.5],
            y_range=[-2.5, 2.5, 0.5],
            x_length=6,
            y_length=6,
            tips=False,
        ).set_stroke(color=colors.BLACK, width=4)

        labels = ax.get_axis_labels().set_color(colors.BLACK).set_stroke(width=1)

        self.circle_points_inner = VGroup()
        self.points_array_inner = []
        for _ in range(20):
            theta = random.random() * 2 * PI
            r = (random.random() + 1) / 2
            x = math.cos(theta) * r
            y = math.sin(theta) * r
            self.points_array_inner.append([x, y])
            point = ax.c2p(x, y)
            self.circle_points_inner.add(Dot(point=point, color=colors.GREEN))

        self.points_array_outer = []
        self.circle_points_outer = VGroup()
        for _ in range(20):
            theta = random.random() * 2 * PI
            r = (random.random() + 1) / 2 + 1
            x = math.cos(theta) * r
            y = math.sin(theta) * r
            self.points_array_outer.append([x, y])
            point = ax.c2p(x, y)
            self.circle_points_outer.add(Dot(point=point, color=colors.RED))

        self.original_space = VGroup(
            ax,
            labels,
            self.circle_points_inner,
            self.circle_points_outer,
        )
        self.play(Write(ax), Write(labels))
        self.play(Write(self.circle_points_inner))
        self.play(Write(self.circle_points_outer))

    def feature_transform(self):
        self.play(self.original_space.animate.move_to([-4, 0, 0]).scale(0.75))

        arrow = (
            Arrow(
                start=self.original_space.get_right(),
                end=self.original_space.get_right() + RIGHT * 3,
                max_tip_length_to_length_ratio=0.05,
            )
            .shift(DOWN)
            .set_stroke(width=4, color=colors.BLACK)
            .set_fill(color=colors.BLACK)
        )

        self.play(Create(arrow))

        transform_tex = MathTex(
            r"""
            r = \sqrt{x^2 + y^2} \\
            \theta = \tan^{-1}\left(\frac{y}{x}\right)
            """,
            color=colors.ORANGE,
            font_size=35,
        ).set_stroke(width=1)

        self.play(Create(transform_tex))
        self.feature_transform_tex = VGroup(arrow, transform_tex)

    def feature_space(self):
        # here, think about duplicating the circle points
        # and then making one travel into its corresponding
        # position in the feature space
        ax = Axes(
            x_range=[-2.5, 2.5, 0.5],
            y_range=[-2.5, 2.5, 0.5],
            x_length=6,
            y_length=6,
            tips=False,
        ).set_stroke(color=colors.BLACK, width=4)

        labels = (
            ax.get_axis_labels(x_label="r", y_label=r"\theta")
            .set_color(colors.BLACK)
            .set_stroke(width=1)
        )
        ax_with_labels = VGroup(ax, labels)

        ax_with_labels.move_to([4, 0, 0]).scale(0.75)
        self.play(Write(ax), Write(labels))

        transformed_inner = self.circle_points_inner.copy()
        animations_inner = []

        def non_linear_transform(x, y):
            return [math.sqrt(x**2 + y**2), math.atan(y / x)]

        for coord, transformed_dot in zip(self.points_array_inner, transformed_inner):
            # I could have just saved the theta and r,
            # but I'll recalculate them for concreteness
            # and as a sanity check

            x, y = coord
            new_coord = ax.c2p(*non_linear_transform(x, y))
            animations_inner.append(transformed_dot.animate.move_to(new_coord))

        transformed_outer = self.circle_points_outer.copy()
        animations_outer = []
        for coord, transformed_dot in zip(self.points_array_outer, transformed_outer):
            x, y = coord
            new_coord = ax.c2p(*non_linear_transform(x, y))
            animations_outer.append(transformed_dot.animate.move_to(new_coord))

        self.play(*animations_inner)
        self.play(*animations_outer)

        # why is the line not showing up!?
        separation_line = Line(
            ax.c2p(1.25, 2.5), ax.c2p(1.25, -2.5), color=colors.BLACK
        )
        self.play(FadeIn(separation_line))
        Axes.c2p
        Axes.coords_to_point
        separation_line_transform = separation_line.copy()
        self.add(separation_line_transform)

        radius = abs(
            self.original_space[0].c2p(1.25, 0)[0] - self.original_space[0].c2p(0, 0)[0]
        )
        circle_separator = Circle(radius=radius, color=colors.BLACK).move_to(
            self.original_space[0].get_origin()
        )
        self.play(Transform(separation_line_transform, circle_separator))
