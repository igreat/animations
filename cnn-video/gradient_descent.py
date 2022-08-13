from manim import *
from video_utils import *


class GradientDescent2d(Scene):
    def construct(self):
        self.camera.background_color = colors.WHITE

        self.gradient_descent_2d()
        self.play(FadeOut(*[mob for mob in self.mobjects]))
        self.wait(2)

    def gradient_descent_2d(self):

        # this is just a sample graph for visualization
        ax = Axes(
            x_range=[-2.5, 2.5, 0.5],
            y_range=[-1, 4, 1],
            x_length=10,
            y_length=7,
            tips=False,
        ).set_stroke(color=colors.BLACK, width=4)

        def func(x):
            return x**4 + 3 * x**3 + x**2 - 3 * x + 1

        def dfunc_dx(x):
            return 4 * x**3 + 9 * x**2 + 2 * x - 3

        graph = ax.plot(func).set_stroke(color=colors.PURPLE)

        labels = ax.get_axis_labels(
            x_label=Text("w", font="Fira Code", weight=BOLD, font_size=30),
            y_label=Text("MSE", font="Fira Code", weight=BOLD, font_size=30),
        ).set_color(colors.BLACK)

        x = ValueTracker(1)

        def get_tangent_line(dot=True):
            def tangent_line():
                y = func(x.get_value())
                line = get_line(
                    [x.get_value(), y], ax, dfunc_dx(x.get_value())
                ).set_stroke(color=colors.RED)
                if dot:
                    tracking_dot = Dot(
                        point=[ax.c2p(x.get_value(), y)], color=colors.BLACK
                    )
                    line = VGroup(line, tracking_dot)
                return line

            return tangent_line

        line = always_redraw(get_tangent_line(dot=True))

        self.play(Write(ax), Write(labels))
        self.play((Write(graph)))
        self.play(Write(line))
        self.play(x.animate.set_value(-2), run_time=3)
        self.play(x.animate.set_value(-0.9))
        self.wait(2)

        # gradient descent algorithm
        temp_text = Text(
            "slope = ",
            font="Fira Code",
            weight=BOLD,
            font_size=30,
            color=colors.BLACK,
        )
        derivative_tex = MathTex(
            r"\frac{\partial \text{MSE}}{\partial w}", color=colors.DARK_RED
        )
        derivative_tex = (
            VGroup(temp_text, derivative_tex).arrange(RIGHT).move_to([4, 2, 0])
        )

        code = """# gradient descent
for _ in range(num_steps):
    w = w - lr * dmse_dw(w)
"""
        gradient_descent_code = Code(
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
        ).move_to([4, 0, 0])

        self.play(Write(derivative_tex))
        self.play(Write(gradient_descent_code))

        alpha = 1.2e-1
        x_anim = x.get_value()
        trace_lines = VGroup()
        start = ax.c2p(x_anim, func(x_anim))
        for _ in range(10):
            x_anim = x_anim - alpha * dfunc_dx(x_anim)
            end = ax.c2p(x_anim, func(x_anim))
            trace_line = Arrow(start, end, color=colors.GREEN, buff=0)
            start = end
            trace_lines.add(trace_line)
            self.play(x.animate.set_value(x_anim), Write(trace_line))


class GradientDescent3d(ThreeDScene):
    def construct(self):
        self.camera.background_color = colors.WHITE
        self.gradient_descent_3d()
        self.wait(2)

    def gradient_descent_3d(self):

        axes = ThreeDAxes(
            x_range=[-1.5, 1.5, 1.5 / 4],
            y_range=[-1.2, 1.2, 1.2 / 4],
            z_range=[-2, 10, 2],
            x_length=15,
            y_length=15,
            z_length=8,
            tips=False,
        ).set_stroke(color=colors.BLACK)
        base_plane = NumberPlane(
            x_range=[-1.51, 1.51, 1.5 / 4],
            y_range=[-1.21, 1.21, 1.2 / 4],
            x_length=15,
            y_length=15,
        ).set_stroke(color=colors.GREEN, opacity=0.5)

        def func(u, v):
            x, y = u, v
            z = (
                4 * x**2
                + x * y
                - 4 * y**2
                - 2.1 * x**4
                + 4 * y**4
                + 1 / 3 * x**6
            )
            return np.array([x, y, 4 * z])

        def dfunc_du(u, v):
            x, y = u, v
            return 4 * (8 * x + y - 8.4 * x**3 + 2 * x**5)

        def dfunc_dv(u, v):
            x, y = u, v
            return 4 * (x - 8 * y + 16 * y**3)

        graph = Surface(
            lambda u, v: axes.c2p(*func(u, v)),
            u_range=[-1.5, 1.5],
            v_range=[-1.2, 1.2],
            resolution=8,
        ).set_color(colors.PURPLE)

        coord = [ValueTracker(1), ValueTracker(0)]

        tracking_dot = Dot3D(color=colors.BLACK, radius=0.03)

        def update_dot(m):
            x, y = coord[0].get_value(), coord[1].get_value()
            m.move_to(axes.c2p(*func(x, y)))

        tracking_dot.add_updater(update_dot)
        tracking_dot.resume_updating()

        full_graph = VGroup(base_plane, axes, graph).scale(0.25)

        self.set_camera_orientation(phi=45 * DEGREES, theta=-10 * DEGREES)
        self.move_camera(frame_center=full_graph.get_center())
        self.play(Write(full_graph))
        self.play(Create(tracking_dot))

        alpha = 5e-3
        x, y = coord[0].get_value(), coord[1].get_value()

        # consider changing this to a tracepath
        trace = TracedPath(
            tracking_dot.get_center, stroke_color=colors.RED, stroke_width=4
        )
        self.begin_ambient_camera_rotation(rate=-PI / 30)
        self.add(trace)
        for _ in range(30):
            x = x - alpha * dfunc_du(x, y)
            y = y - alpha * dfunc_dv(x, y)
            self.play(
                coord[0].animate.set_value(x),
                coord[1].animate.set_value(y),
            )
            print(func(x, y)[2])
        self.stop_ambient_camera_rotation()
