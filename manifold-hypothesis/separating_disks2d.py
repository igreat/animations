from manim import *
import torch
import colors
import numpy as np
from utils import generate_outer_inner_circles
from models import DiskClassifier2D
from modules import Grid

config.background_color = colors.WHITE

class SeparatingDisks2D(Scene):
    def construct(self):
        self.separate_disks()
        self.wait()

    def separate_disks(self):
        pass
        