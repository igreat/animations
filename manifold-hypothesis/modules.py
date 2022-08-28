from manim import *
import colors


class Grid(VGroup):
    def __init__(self, axes, x_range, y_range, *vmobjects, **kwargs):
        super().__init__(*vmobjects, **kwargs)
        self.submobjects = VGroup()
        self.grid_2d = []
        positions = []
        y_range[1] += y_range[2]
        x_range[1] += x_range[2]
        for y in np.arange(*y_range):
            row = []
            for x in np.arange(*x_range):
                point = Dot(axes.c2p(x, y), 0)
                row.append(point)
                self.submobjects.add(point)
                positions.append([x, y])

            self.grid_2d.append(row)

        self.grid_lines = VGroup()

        for i in range(len(self.grid_2d[0])):
            for j in range(len(self.grid_2d[1]) - 1):

                line = Line(color=colors.BLACK)
                line.add_updater(
                    lambda m, dt, i=i, j=j: m.set_points_by_ends(
                        self.grid_2d[i][j].get_center(),
                        self.grid_2d[i][j + 1].get_center(),
                    )
                )
                self.grid_lines.add(line)

        self.array = np.array(positions)

        for j in range(len(self.grid_2d[1])):
            for i in range(len(self.grid_2d[0]) - 1):

                line = Line(color=colors.BLACK)
                line.add_updater(
                    lambda m, dt, i=i, j=j: m.set_points_by_ends(
                        self.grid_2d[i][j].get_center(),
                        self.grid_2d[i + 1][j].get_center(),
                    )
                )
                self.grid_lines.add(line)


class AdaptiveAxes:
    pass
