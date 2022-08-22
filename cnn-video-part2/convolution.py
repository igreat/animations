from webbrowser import get
from manim import *
from video_utils import *
import colors


class Convolution(ThreeDScene):
    def construct(self):
        self.camera.background_color = colors.WHITE
        # try to understand why UP DOWN RIGHT LEFT seem to be screwed up here
        # maybe the rotations aren't what they seem to be?
        self.set_camera_orientation(phi=1 / 9 * TAU, theta=-(1 / 4 + 1 / 16) * TAU)

        # scaling is messed up for non square inputs and kernels
        input_size = (5, 5)
        kernel_size = (3, 3)

        output_size = (
            input_size[0] - kernel_size[0] + 1,
            input_size[1] - kernel_size[1] + 1,
        )

        orange_array = get_color_array(input_size, colors.DESERT)
        black_array = get_color_array(kernel_size, colors.BLACK)
        green_array = get_color_array(output_size, colors.GREEN)

        input_array = np.random.randint(0, 255, (5, 5))
        input_channel = get_pixels_grid_with_nums(
            orange_array, image_nums=input_array, border_width=4
        ).shift(IN * 3)
        input_channel.z_index = 0

        output_channel = get_pixels_grid(green_array).set_stroke(width=4).shift(OUT * 3)
        output_channel.z_index = 3

        kernel = (
            get_pixels_grid(black_array)
            .set_stroke(width=4)
            .move_to(input_channel.get_corner(UL))
            .shift(DOWN * kernel_size[1] / 2, RIGHT * kernel_size[0] / 2)
            .set_opacity(0.5)
        )
        kernel.z_index = 1

        input_highlight = Square(1, color=colors.BLACK, fill_opacity=0.5).move_to(
            output_channel[0].get_center()
        )
        input_highlight.z_index = 5

        connection_lines = build_connection_lines(
            kernel, input_highlight, color=colors.DARK_RED
        )
        connection_lines.z_index = 2

        all_mobs = VGroup(
            input_channel, output_channel, kernel, input_highlight, connection_lines
        )

        scale_factor = 0.8
        all_mobs.scale(scale_factor)

        self.play(Write(all_mobs))

        kernel_and_highlight = VGroup(kernel, input_highlight)

        for i in range(output_size[0]):

            for _ in range(output_size[1] - 1):
                # shifting horizontally
                self.play(
                    kernel_and_highlight.animate.shift(RIGHT * scale_factor),
                )
            # shifting vertically
            if i != output_size[0] - 1:
                self.play(
                    kernel_and_highlight.animate.shift(
                        LEFT * (output_size[1] - 1) * scale_factor, DOWN * scale_factor
                    ),
                )

        self.wait(2)
