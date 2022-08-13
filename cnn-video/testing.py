from manim import *
from video_utils import *


def rotate_cube(cube: VMobject, phi: float, theta: float) -> VMobject:
    pass


class Testing3D(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#FDF6E3"

        img = ImageMobject("assets/bird4")
        cube = Cube(side_length=2)
        # self.set_camera_orientation(phi=-1 / 12 * TAU, theta=-1 / 12 * TAU)

        self.play(FadeIn(cube))
        self.play(cube.animate.shift(LEFT * 2))
        self.wait()


class AverageImage(Scene):
    def construct(self):
        dogs = [f"assets/dog-images/dog{i}.jpg" for i in range(1, 11)]
        dog_avg = get_average_image(dogs)
        img = ImageMobject(dog_avg)
        img.height = 7

        svg_img = SVGMobject(
            "assets/dog-rough-template.svg", fill_opacity=0
        ).set_stroke(width=3, color="#333333")
        svg_img.height = 7

        self.play(FadeIn(img))
        self.play(Write(svg_img))
        self.wait(2)
