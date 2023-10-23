from pathlib import Path

# Torch imports
import torch
from torch import Tensor, nn


class Mlp(nn.Module):
    """
    This is main class for the neural network model.
    It takes in an input dim and a list of hidden dims and generates a neural network.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        """
        Output dim is the last dim in `hidden_dims`.
        """

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Creating layers out of hidden dims
        layers = []
        cur_dim = input_dim
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(cur_dim, hidden_dims[i]))
            layers.append(nn.GELU())
            cur_dim = hidden_dims[i]
        # No activation after last layer
        layers.append(nn.Linear(cur_dim, hidden_dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """
        Returns an output after passing the input through the model.
        """
        x = self.layers(x)
        return x


def load_model(model: Mlp, path_to_model: Path):
    """
    Load model to cpu for use.
    """

    state_dict = torch.load(path_to_model, map_location="cpu")
    model.load_state_dict(state_dict)