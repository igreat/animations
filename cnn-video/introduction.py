from manim import *
import cv2
from video_utils import *
from PIL import Image
from pytorch_utils.feature_extraction import get_deep_activations, get_edges


class IntroductionPart1(Scene):
    def construct(self):
        self.camera.background_color = colors.WHITE

        image_array = cv2.imread("assets/dog-grayscale.jpg")
        image = ImageMobject(image_array)

        image.height = 4
        image.shift(UP * 1.5)

        self.play(FadeIn(image))

        pixels = image_array.shape[0]
        k = 8
        k_actual = image.height / pixels * k

        kernel_actual = Square(k_actual).set_stroke(color=colors.PURPLE, width=2)
        kernel_actual.move_to(image.get_center())

        # getting a sample 8x8 portion of the image
        p = pixels // 2
        sample_array = image_array[p : p + 5, p : p + 5]
        kernel_zoomed_inner = get_pixels_grid_with_nums(
            sample_array,
            border_width=2,
            nums_color=colors.WHITE,
            # nums_stroke_width=0.1,
            # nums_stroke_color=colors.BLACK,
        ).scale(0.5)

        # giving it a border
        kernel_border = Square(2.55).set_stroke(color=colors.RED, width=4)

        kernel_border.move_to(kernel_zoomed_inner.get_center())
        kernel_zoomed = VGroup(kernel_border, kernel_zoomed_inner)

        kernel_zoomed.shift(DOWN * 2)

        connection_lines = build_connection_lines(kernel_actual, kernel_zoomed)
        connection_lines.z_index = 2
        dog_and_kernel = Group(image, kernel_actual)

        self.play(DrawBorderThenFill(kernel_actual), run_time=1)
        self.add(connection_lines)
        self.play(
            GrowFromPoint(kernel_zoomed, kernel_actual.get_center()),
            run_time=1,
        )

        self.play(
            dog_and_kernel.animate.scale(0.8).shift(LEFT * 4),
            kernel_zoomed.animate.scale(0.8).shift(LEFT * 4.5),
        )
        self.wait()

        # arrow
        arrow1 = (
            Arrow(start=LEFT, end=RIGHT, color=colors.BLACK)
            .scale(0.5, scale_tips=True)
            .shift(LEFT * 1.5)
        )
        self.play(Write(arrow1))

        # processing machine
        gear = (
            SVGMobject("assets/gear.svg", fill_color=colors.ORANGE)
            .shift(UP * 0.8)
            .set_stroke(color=colors.BLACK, width=3.5)
        )
        box = (
            Rectangle(height=4, width=3)
            .set_stroke(color=colors.BLACK, width=5)
            .shift(UP * 0.3)
        )
        text = (
            Text(
                "processing...",
                font_size=22,
                font="Fira Code",
                color=colors.BLACK,
                weight=BOLD,
            )
            .move_to(box.get_bottom())
            .shift(UP)
        )
        machine = VGroup(box, gear, text).shift(RIGHT * 0.5)
        self.play(Write(machine), run_time=2)

        arrow2 = arrow1.copy().shift(RIGHT * 4)
        self.play(Write(arrow2))

        dog_text = Text(
            '"dog"', font_size=60, font="Fira Code", color=colors.BLACK
        ).shift(RIGHT * 4.5, DOWN * 0.1)
        self.play(Write(dog_text))
        self.wait(2)

        self.play(
            Unwrite(machine),
            Uncreate(kernel_zoomed),
            Uncreate(kernel_actual),
            Unwrite(dog_text),
            Unwrite(connection_lines),
            Unwrite(arrow1),
            Unwrite(arrow2),
            run_time=2,
        )

        self.play(image.animate.move_to(ORIGIN).scale(1.75))

        dogs = [f"assets/dog-images/dog{i}.jpg" for i in range(1, 11)]
        dog_avg = get_average_image(dogs)
        dog_avg_img = ImageMobject(dog_avg).set_opacity(0.9)
        dog_avg_img.height = image.height

        self.play(FadeIn(dog_avg_img))
        self.play(
            image.animate.shift(2.5 * LEFT),
            dog_avg_img.animate.shift(2.5 * RIGHT),
            run_time=0.5,
        )
        self.wait()
        self.play(
            image.animate.shift(2.5 * RIGHT),
            dog_avg_img.animate.shift(2.5 * LEFT),
            run_time=0.5,
        )

        self.play(FadeOut(dog_avg_img))

        dog_outlines = (
            SVGMobject("assets/dog-rough-template.svg", fill_opacity=0)
            .set_stroke(width=4, color=colors.BLACK)
            .shift(LEFT * 0.25, DOWN * 0.25)
        )
        dog_outlines.height = image.height

        self.play(Write(dog_outlines), image.animate.set_opacity(0.5))
        self.wait()

        self.play(Unwrite(dog_outlines), image.animate.set_opacity(1))

        self.wait(2)


class IntroductionPart2(Scene):
    def construct(self):
        self.camera.background_color = colors.WHITE

        # I keep a copy of the Image object since that's
        # easier to work with pytorch
        bird_pil = Image.open("assets/bird.jpg")
        bird_image = ImageMobject(bird_pil)
        bird_image.height = 5

        bird_edges = get_edges(bird_pil)

        # picking num_images sample edge images to display
        num_images = 6
        edges_group = Group()
        for bird_edge in bird_edges[6 : num_images + 6]:
            bird_edge = np.array(bird_edge * 255, dtype=np.uint8)
            edges_group.add(ImageMobject(bird_edge))

        edges_group.height = 20 / num_images - 1.2
        edges_group.arrange_in_grid(cols=2, buff=0.1)

        deep_activations = get_deep_activations(bird_pil)
        deep_activations_group = Group()
        for activation in deep_activations[12 : num_images + 12]:
            activation = np.array(activation * 255, dtype=np.uint8)
            deep_activations_group.add(ImageMobject(activation))

        deep_activations_group.height = 20 / num_images - 1.2
        deep_activations_group.arrange_in_grid(cols=2, buff=0.1)

        self.play(FadeIn(bird_image))
        self.play(bird_image.animate.shift(LEFT * 4))
        self.play(GrowFromPoint(edges_group, bird_image.get_center()))
        self.play(
            FadeIn(deep_activations_group),
            deep_activations_group.animate.shift(RIGHT * 4),
        )

        all_mobs = Group(bird_image, edges_group, deep_activations_group)
        self.play(all_mobs.animate.shift(DOWN * 0.5))

        text = (
            Text(
                "feature hierarchies",
                color=colors.BLACK,
                font="Fira Code",
                weight=BOLD,
                font_size=40,
            )
            .move_to(all_mobs.get_top())
            .shift(UP * 0.6)
        )
        self.play(Write(text))
        self.wait(2)
