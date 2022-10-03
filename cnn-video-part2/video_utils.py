from manim import *
import numpy as np
from PIL import Image
import colors

# consider generalizing this to also rgb images
def get_pixels_grid(
    image_array: np.uint8,
    border_color=colors.BLACK,
    three_d=False,
) -> VGroup:

    # returns a grid of pixels representing the image_array as a VGroup
    # image_array: np.uint8 of shape (height, width, 3)

    columns = image_array.shape[1]

    pixel_grid = VGroup()
    for cols in image_array:
        for pixel_color in cols:

            square = Square(
                side_length=1,
                color=rgb_to_color(pixel_color / 255),
                fill_opacity=1,
                shade_in_3d=three_d,
            ).set_stroke(color=border_color, width=1)

            pixel_grid.add(square)

    # figure out why I have to put in columns here instead
    # of rows to actually get the correct number of rows
    pixel_grid.arrange_in_grid(rows=columns, buff=0)

    return pixel_grid


def get_pixels_grid_with_nums(
    image_array: np.uint8,
    image_nums=None,
    border_color=colors.BLACK,
    three_d=False,
    border_width=1,
    nums_stroke_width=0,
    nums_color=colors.BLACK,
    nums_stroke_color=colors.BLACK,
) -> VGroup:
    # this will simply display the average pixel value at the center of the pixel
    # image_array: np.uint8 of shape (rows, columns, 3)

    pixel_grid = get_pixels_grid(image_array, border_color, three_d).set_stroke(
        width=border_width
    )

    num_rows = image_array.shape[0]
    font_size = 120 / num_rows
    if image_nums is None:
        # setting it to the average pixels if not specified
        image_nums = image_array.mean(axis=2).astype(np.uint8)

    labels = [
        Text(
            str(x), font="Fira Code", font_size=font_size, weight=BOLD, color=nums_color
        )
        .move_to(square.get_center())
        .set_stroke(width=nums_stroke_width, color=nums_stroke_color)
        for square, x in zip(pixel_grid, image_nums.flatten())
    ]
    pixel_grid_with_nums = VGroup(pixel_grid, *labels)

    return pixel_grid_with_nums


def get_average_image(images: list[str]) -> np.uint8:
    images = [Image.open(image) for image in images]

    # for simplicity, I will crop the images to be squares and the length
    # of those squares will be the minimum pixels in row or column
    min_dim = min([min(image.size) for image in images])
    images = [
        np.asarray(center_crop(resize_image(image, min_dim), min_dim))
        for image in images
    ]

    return np.array(images).mean(axis=0).astype(np.uint8)


# change this to center resize
def center_crop(image: Image, length: int) -> Image:
    width, height = image.size

    assert (
        height >= length and width >= length
    ), "crop size greater than image resolution"

    half = length // 2

    mid_x, mid_y = width // 2, height // 2
    image = image.crop((mid_x - half, mid_y - half, mid_x + half, mid_y + half))
    return image


def resize_image(image: Image, length: int) -> Image:
    # returns a resized image with maintained aspect ratio

    width, height = image.size
    wh_ratio = width / height

    if width <= height:
        width = length
        height = width / wh_ratio
    else:
        height = length
        width = height * wh_ratio

    image = image.resize((int(width), int(height)), Image.ANTIALIAS)
    return image


def get_color_array(size, color=colors.WHITE) -> np.uint8:
    # returns an array of shape (height, width, 3) representing a single color
    array = np.ones((*size, 3)) * np.array(hex_to_rgb(color)) * 255
    return array.astype(np.uint8)


def build_connection_lines(
    mob1: VMobject, mob2: VMobject, color=colors.RED, dotted=True, stuck=True
) -> VMobject:
    directions = [UR, DR, UL, DL]

    def get_connection_lines():
        connection_lines = VGroup()
        for start, end in zip(
            [mob1.get_corner(d) for d in directions],
            [mob2.get_corner(d) for d in directions],
        ):
            if dotted:
                connection_lines.add(DashedLine(start, end, color=color))
            else:
                connection_lines.add(Line(start, end, color=color))

        return connection_lines

    if stuck:
        return always_redraw(get_connection_lines)

    return get_connection_lines()


def build_layer_lines(
    layer1: VGroup,
    layer2: VGroup,
    opacities,
    colors=[GREEN, RED],
    dotted=False,
    stuck=True,
    start=ORIGIN,
    end=ORIGIN,
    always_back=False,
    inflate_opacities=1,
) -> VGroup:
    # this builds a single line between two mobjects

    # colors should only have two elements

    def get_layer_lines():
        connection_lines = VGroup()
        i = 0
        for mob1 in layer1:
            point1 = mob1.get_critical_point(start)
            for mob2 in layer2:
                point2 = mob2.get_critical_point(end)
                # print(colors[i].get_center())
                opacity = opacities[i]
                if type(opacity) == ValueTracker:
                    opacity = opacity.get_value()

                if opacity > 0:
                    color = colors[0]
                else:
                    color = colors[1]

                if dotted:
                    connection_lines.add(
                        DashedLine(point1, point2, color=color).set_stroke(
                            opacity=clip(abs(inflate_opacities * opacity), 0, 1)
                        )
                    )
                else:
                    connection_lines.add(
                        Line(point1, point2, color=color).set_stroke(
                            opacity=clip(abs(inflate_opacities * opacity), 0, 1)
                        )
                    )
                i += 1
        if always_back:
            connection_lines.z_index = 0
        return connection_lines

    if stuck:
        return always_redraw(get_layer_lines)

    return get_layer_lines()


def get_color_proportion(
    color1: np.ndarray, color2: np.ndarray, ratio: float
) -> np.ndarray:
    # ratio represents the how much of color1
    # there should be with respect to color2

    return (1 - ratio) * color1 + ratio * color2


def clip(x: float, min: float, max: float) -> float:
    # a scalar clip function
    if x > max:
        return max
    if x < min:
        return min
    return x


def get_line(point, axes, slope):
    # (y2 - y1) / (x2 - x1) = slope
    return axes.plot(lambda x: slope * (x - point[0]) + point[1])
