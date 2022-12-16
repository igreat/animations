# TODO: complete the manifold hypothesis scene here
#       1. show examples of real world datasets like tabular data
#          and handwritten digit pictures as a 28 by 28 grid
#
#       2. do some kind of visualization showing how mnist
#          is separated in some n dimentional space, but use
#          dimentionality reduction like PCA to bring it 3D
#          for visualization purposes

from manim import *
import colors
import numpy as np
from utils import *
from modules import *
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# train the model elsewhere and then import it here
# only visualization will happen here

config.background_color = colors.WHITE

# load mnist dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


full_loader = DataLoader(
    datasets.MNIST("data", train=False, download=True, transform=transform),
    batch_size=64,
    shuffle=True,
)
image_loader = DataLoader(
    datasets.MNIST("data", train=False, transform=transforms.ToTensor()),
    batch_size=1,
    shuffle=True,
)

# load the model
model = torch.load("saved_models/mnist_model.pt")


class ManifoldHypothesis(ThreeDScene):
    def construct(self):
        self.datasets_examples()
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
        table_title.next_to(table_object, UP, buff=0.5)

        table = Group(table_object, table_title)

        # display an mnist image as an ImageMobject in manim

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

    def mnist_separation(self):
        """
        To make the mnist digits face the camera, we need to match the
        camera's rotation speed with the rotation speed of the mnist digits
        """
        pass
