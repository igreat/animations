from manim import *
import colors
from functools import partial


class Grid(VGroup):
    def __init__(self, axes, x_range, y_range, lattice_radius=0, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        self.submobjects = VGroup()
        self.grid_2d = []
        positions = []
        y_range[1] += y_range[2]
        x_range[1] += x_range[2]
        for y in np.arange(*y_range):
            row = []
            for x in np.arange(*x_range):
                point = Dot(axes.c2p(x, y), lattice_radius, color=colors.BLACK)
                row.append(point)
                self.submobjects.add(point)
                positions.append([x, y])

            self.grid_2d.append(row)

        self.grid_lines = VGroup()

        def update_grid_connection(m, pos1, pos2):
            x1, y1 = pos1
            x2, y2 = pos2
            return m.set_points_by_ends(
                self.grid_2d[y1][x1].get_center(),
                self.grid_2d[y2][x2].get_center(),
            )

        # building the horizontal lines
        for i in range(len(self.grid_2d[0])):
            for j in range(len(self.grid_2d) - 1):
                line = Line(color=colors.BLACK)
                update_line = partial(
                    update_grid_connection, pos1=[i, j], pos2=[i, j + 1]
                )
                line.add_updater(update_line)
                self.grid_lines.add(line)

        self.array = np.array(positions)
        # building the vertical lines
        for j in range(len(self.grid_2d)):
            for i in range(len(self.grid_2d[0]) - 1):
                line = Line(color=colors.BLACK)
                update_line = partial(
                    update_grid_connection, pos1=[i, j], pos2=[i + 1, j]
                )
                line.add_updater(update_line)
                self.grid_lines.add(line)


class MnistImage:
    def __init__(
        self,
        image: np.ndarray,
        position: np.ndarray,
        height=0.5,
    ):

        self.image = ImageMobject(image)
        self.image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        self.image.height = height
        self.position_dot = Dot(
            position, radius=0
        )  # representing the position as a dot for convenience
        self.image.move_to(position)
        self.image.add_updater(lambda m, dt: m.move_to(self.position_dot))

    def fix_angle(self, camera):
        # I actually have no idea why these initial numbers work
        self.image.theta = -90 * DEGREES
        self.image.phi = 75 * DEGREES

        def match_angle(mob):
            mob.rotate(-mob.theta, OUT)
            mob.rotate(-mob.phi, UP)
            mob.rotate(camera.theta, OUT)
            mob.rotate(camera.phi, UP)
            mob.theta = camera.theta
            mob.phi = camera.phi

        self.image.add_updater(lambda m, dt: match_angle(m))
        self.image.resume_updating()
        return self
