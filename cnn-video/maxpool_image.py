from manim import *
import numpy as np
from video_utils import *


class MaxpoolImage(Scene):
    def construct(self):

        self.camera.background_color = colors.WHITE

        img = ImageMobject("assets/bird4")
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["box"])
        img.height = 7
        img_array = img.get_pixel_array()[:, :, :3]

        # number of pixels in each row
        pixels = img_array.shape[0]
        # kernel size
        k = 4
        assert (
            pixels % k == 0
        ), f"kernel size {k} must divide pixel count {pixels} assuming square image"

        k_actual = img.height / pixels * k

        kernel = Square(k_actual)
        kernel.align_to(img, direction=[-1, 1, 0])
        self.play(FadeIn(img))
        self.play(FadeIn(kernel))
        for j in range(pixels // k):
            for i in range(pixels // k):

                # doing max pool on each color channel separately
                img_array[k * j : k * (j + 1), k * i : k * (i + 1), 0] = np.max(
                    img_array[k * j : k * (j + 1), k * i : k * (i + 1), 0]
                )
                img_array[k * j : k * (j + 1), k * i : k * (i + 1), 1] = np.max(
                    img_array[k * j : k * (j + 1), k * i : k * (i + 1), 1]
                )
                img_array[k * j : k * (j + 1), k * i : k * (i + 1), 2] = np.max(
                    img_array[k * j : k * (j + 1), k * i : k * (i + 1), 2]
                )

                img.pixel_array[:, :, :3] = img_array
                # self.wait(0.1)
                if i == 32 // k - 1:
                    break
                self.play(kernel.animate.shift(RIGHT * k_actual), run_time=0.2)

            if j == 32 // k - 1:
                break
            self.play(
                kernel.animate.shift(DOWN * k_actual).shift(
                    LEFT * k_actual * (32 // k - 1)
                ),
                run_time=0.2,
            )

        self.wait(2)
