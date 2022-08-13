import torch
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import numpy as np


def get_edges(image: Image) -> torch.Tensor:
    # returns a numpy array representing varius edge images

    model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
    model.requires_grad_(False)

    feature_extractor = create_feature_extractor(
        model,
        return_nodes=["layer1.0.conv1"],
    )

    image = pil_to_tensor(image) / 255.0

    # giving it a "fake" batch dimention as model expects
    image.unsqueeze_(0)

    activations = feature_extractor(image)
    edges = activations["layer1.0.conv1"].squeeze(0)

    return edges


def get_deep_activations(image: Image) -> torch.Tensor:
    # returns a numpy array representing varius edge images

    model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
    model.requires_grad_(False)

    feature_extractor = create_feature_extractor(
        model,
        return_nodes=["layer1.2.relu"],
    )

    image = pil_to_tensor(image) / 255.0

    # giving it a "fake" batch dimention as model expects
    image.unsqueeze_(0)

    activations = feature_extractor(image)
    targets = activations["layer1.2.relu"].squeeze(0)

    return targets
