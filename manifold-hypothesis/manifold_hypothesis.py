from manim import *
import colors
import numpy as np
from utils import *
from modules import *
from models import MnistClassifier
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

config.background_color = colors.WHITE

# load mnist dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


full_loader = DataLoader(
    datasets.MNIST("data", train=False, download=True, transform=transform),
    batch_size=1000,
    shuffle=True,
)
image_loader = DataLoader(
    datasets.MNIST("data", train=False, transform=transforms.ToTensor()),
    batch_size=1,
    shuffle=True,
)

# load the model
mnist_feature_extractor = MnistClassifier()
mnist_feature_extractor.load_state_dict(torch.load("saved_models/mnist_model_tanh.pt"))
mnist_feature_extractor.eval().requires_grad_(False)


class ManifoldHypothesis(ThreeDScene):
    def construct(self):
        # self.datasets_examples()
        # self.manifold_examples()
        self.mnist_classifier_code()
        # self.mnist_separation()
        self.wait()

    def datasets_examples(self):

        # TODO: add color to the table
        table_object = Table(
            [
                ["PassengerId", "Survived", "Pclass", "Sex", "Age"],
                ["1", "0", "3", "male", "22"],
                ["2", "1", "1", "female", "38"],
                ["3", "1", "3", "female", "26"],
                ["4", "1", "1", "female", "35"],
                ["5", "0", "3", "male", "35"],
            ],
            include_outer_lines=True,
        ).set_color(color=colors.BLACK)
        table_object.height = 4
        table_object.get_entries().set_stroke(width=1)

        table_title = (
            Text("Titanic Dataset", color=colors.BLACK)
            .set_stroke(color=colors.BLACK, width=1)
            .scale(0.5)
        )
        table_title.next_to(table_object, UP, buff=0.5).scale(1.5)

        table = Group(table_object, table_title)

        # get the first image
        data, _ = next(iter(image_loader))
        image = data.numpy()

        # remove batch dimension and convert to uint8 and range [0, 255]
        image = np.uint8(np.squeeze(image) * 255)
        image = ImageMobject(image).shift(IN * 0.5)
        # somehow shifting it fixes a bug where it doesn't show up immediately

        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        # scale the image
        image.height = 4
        # show the image

        examples = Group(table, image)
        examples.arrange(RIGHT, buff=0.5)

        self.play(FadeIn(table))
        self.play(FadeIn(image))
        self.wait(2)

        self.play(
            table.animate.shift(UP * 1.2 + LEFT * 0.25).scale(0.75),
            image.animate.shift(UP * 1.2 + LEFT * 0.25).scale(0.75),
        )

        # question mark for effect
        question_mark = (
            Text("?", color=colors.DARK_RED)
            .set_stroke(width=1, color=colors.DARK_RED)
            .scale(4.5)
        )
        question_mark.next_to(examples, DOWN, buff=0.5)
        self.play(Write(question_mark))
        self.wait(2)
        self.play(Unwrite(question_mark))
        self.wait()

        manifold1 = (
            Torus(checkerboard_colors=[colors.RED, colors.DARK_RED])
            .scale(0.5)
            .rotate(-80 * DEGREES, RIGHT)
            .next_to(image, DOWN, buff=0.6)
        )
        manifold2 = (
            Surface(
                lambda u, v: np.array([u, v, np.sin(u**2 + v**2)]),
                u_range=[-1, 1],
                v_range=[-1, 1],
                checkerboard_colors=[colors.DESERT, colors.ORANGE],
            )
            .rotate(-80 * DEGREES, RIGHT)
            .scale(2)
            .next_to(table, DOWN, buff=0.5)
        )
        self.play(FadeIn(manifold1))
        self.play(FadeIn(manifold2))
        self.wait()

        self.play(FadeOut(Group(table, image, manifold1, manifold2)))

    def manifold_examples(self):
        """
        Here I will present a few examples of manifolds in different dimensions
        """

        ### present a one dimentional manifold inside 2d space ###
        axes2d = Axes(
            x_range=[-1, 1, 0.2],
            y_range=[-1, 1, 0.2],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": False},
        ).set_stroke(color=colors.BLACK, width=2)
        axes2d_bounding_box = SurroundingRectangle(axes2d, buff=0).set_stroke(
            color=colors.BLACK, width=2
        )

        line = axes2d.plot(lambda x: x).set_stroke(color=colors.PURPLE, width=3)

        circle = (
            Circle(radius=3)
            .next_to(axes2d_bounding_box, ORIGIN)
            .set_stroke(color=colors.RED, width=3)
        )

        full_axes2d = VGroup(axes2d, axes2d_bounding_box)
        full_1d_manifold = VGroup(full_axes2d, line, circle)

        self.play(Write(full_axes2d))
        self.wait()
        self.play(Write(line))
        self.play(Write(circle))
        self.wait()

        self.play(full_1d_manifold.animate.shift(LEFT * 5).scale(0.5))

        ### present a 2 dimensional manifold ###
        axes3d = ThreeDAxes(
            x_range=[-1, 1, 0.2],
            y_range=[-1, 1, 0.2],
            z_range=[-1, 1, 0.2],
            x_length=6,
            y_length=6,
            z_length=6,
            axis_config={"include_tip": False},
        ).set_stroke(color=colors.BLACK, width=2)
        axes3d.rotate(30 * DEGREES, UP).rotate(-30 * DEGREES, RIGHT)

        axes3d_bounding_box = (
            SurroundingRectangle(axes3d, buff=0.25)
            .set_stroke(color=colors.BLACK, width=2)
            .shift(DL * 0.1)
        )

        plane = (
            Square(5)
            .rotate(120 * DEGREES, RIGHT)
            .next_to(axes3d_bounding_box, ORIGIN)
            .set_fill(opacity=0.25, color=colors.DESERT)
            .set_stroke(width=4, color=colors.DESERT)
            .shift(DOWN)
        )

        sphere = (
            Sphere(
                ORIGIN, radius=1.25, checkerboard_colors=[colors.PURPLE, colors.PURPLE]
            )
            .rotate(90 * DEGREES, RIGHT)
            .shift(UP)
            .set_opacity(0.5)
        )

        axes3d.z_index = 3
        sphere.z_index = 1
        plane.z_index = 0

        full_2d_manifold = VGroup(axes3d, axes3d_bounding_box, plane, sphere)

        self.play(Write(VGroup(axes3d, axes3d_bounding_box)))
        self.play(Write(plane))
        self.play(FadeIn(sphere))
        self.wait()
        self.play(
            full_2d_manifold.animate.scale_to_fit_height(axes2d_bounding_box.height)
        )
        self.wait()

        ### a 0 dimensional manifold for the sake of completeless ###

        numberline = NumberLine(
            x_range=[-1, 1, 0.5], length=axes2d_bounding_box.width
        ).set_stroke(width=2, color=colors.BLACK)
        numberline.set_x(-axes2d.get_x())
        axes1d_bounding_box: RoundedRectangle = axes2d_bounding_box.copy()
        axes1d_bounding_box.next_to(numberline, ORIGIN)

        # dot
        dot = (
            Dot(numberline.get_center(), radius=0.04)
            .set_stroke(width=2, color=colors.RED)
            .set_fill(color=colors.RED)
            .shift(LEFT * 0.1)
        )

        self.play(Write(VGroup(numberline, axes1d_bounding_box)))
        self.play(Write(dot))
        self.wait()

        full_0d_manifold = VGroup(numberline, axes1d_bounding_box, dot)
        self.play(FadeOut(VGroup(full_0d_manifold, full_1d_manifold, full_2d_manifold)))

    def mnist_classifier_code(self):
        """
        Here, I'll show the code I used to make the neural network as
        a cheap way of showcasing what neural network structure I'm using
        """
        code = r"""import torch
from torch import nn

class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        # fully connected linear layers (FC)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)
"""
        model_code = Code(
            code=code,
            tab_width=2,
            background="rectangle",
            language="Python",
            font="Menlo, Monaco",
            style="vs",
            background_stroke_color=colors.BLACK,
            background_stroke_width=4,
            line_spacing=1,
            font_size=18,
        )
        self.play(Write(model_code))
        self.wait()
        self.play(Unwrite(model_code))
        self.wait()

    def mnist_separation(self):
        """
        To make the mnist digits face the camera, we need to match the
        camera's rotation speed with the rotation speed of the mnist digits
        """

        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        # build the threed axis
        axes = (
            ThreeDAxes(
                x_range=[-1, 1],
                y_range=[-1, 1],
                z_range=[-1, 1],
                x_length=8,
                y_length=8,
                z_length=6,
                axis_config={"include_tip": False},
            )
            .set_color(color=colors.BLACK)
            .set_stroke(width=1)
        )

        self.camera
        self.begin_ambient_camera_rotation(-0.1)

        self.play(Write(axes))

        data, _ = next(iter(full_loader))

        data = data.view(-1, 28 * 28).requires_grad_(False)
        all_features = mnist_feature_extractor(data)
        # perform PCA on all layers and gather only first n images
        n = 100
        reduced_features = [reduce_dimentionality(x, 3)[:n] for x in all_features]

        # normalizing the features to a range of [-1 to 1] for convenience
        reduced_normalized_features = []
        for feature in reduced_features:
            # making the range 0 to 1
            feature = (feature - feature.min()) / (feature.max() - feature.min())

            # now making the range -1 to 1
            feature = (feature - 0.5) * 2
            reduced_normalized_features.append(feature)

        # initialize the mnist images
        mnist_images: list[MnistImage] = []

        # here for the image we must deprocess the image since it's been normalized
        for image_array, initial_position in zip(
            all_features[0], reduced_normalized_features[0]
        ):
            # mean=0.1307, std=0.3081
            image_array = np.uint8((image_array.view(28, 28) * 0.3081 + 0.1307) * 255)
            mnist_images.append(
                MnistImage(
                    image=image_array, position=axes.c2p(*initial_position)
                ).fix_angle(self.camera)
            )

        images: list[ImageMobject] = [m.image for m in mnist_images]
        animations = [FadeIn(image) for image in images]
        self.play(*animations)

        # showing the transformations
        for batch_features in reduced_normalized_features[1:]:
            animations = []
            for image, feature in zip(mnist_images, batch_features):
                animations.append(
                    image.position_dot.animate.move_to(axes.c2p(*feature))
                )

            self.play(*animations)
            self.wait()

        # TODO: maybe increase the depth of the network as that's an easy variable to change
        #       must retrain the model though
