from turtle import fillcolor
from manim import *
from PIL import Image
import colors


class ConvolutionLayer(ThreeDScene):
    def construct(self):
        self.camera.background_color = colors.WHITE
        self.flatten_vs_spatial()

    def flatten_vs_spatial(self):

        bird_pil = Image.open("assets/bird.png").resize((32, 32))
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

        image_pixels.arrange_in_grid(cols=32, buff=0)
        image_pixels.height = 5
        image_pixels.z_index = 0

        flattened_image = (
            Prism([12, 1, 1])
            .scale(0.3)
            .set_fill(color=colors.DESERT)
            .set_stroke(width=1, color=colors.BLACK)
        )
        flat_dim_tex = (
            MathTex(r"3072", color=colors.BLACK, font_size=30)
            .set_stroke(width=1)
            .next_to(flattened_image, DOWN)
        )

        one = (
            MathTex(r"1", color=colors.BLACK, font_size=30)
            .set_stroke(width=1)
            .next_to(flattened_image, LEFT)
        )

        labeled_flat = VGroup(flattened_image, flat_dim_tex, one)
        flattened_image.z_index = 1

        self.play(FadeIn(image_pixels))
        self.wait()
        self.play(image_pixels.animate.arrange(RIGHT, buff=0), run_time=2)
        self.play(FadeTransform(image_pixels, flattened_image))
        self.remove(image_pixels)
        self.add(flattened_image)
        self.play(FadeIn(flat_dim_tex, shift=UP), FadeIn(one, shift=RIGHT))
        self.play(labeled_flat.animate.shift([-3, 0, 0]))
        self.wait(2)

        
