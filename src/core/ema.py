"""
EMA (Exploratory Model Analysis) utilities.
"""
# Standard
from typing import Optional

# Machine Learning
import torch


# Print model stats.
def print_model_stats(
    model: torch.nn.Module,
    model_name: Optional[str] = None,
) -> None:
    """Print model statistics."""
    num_parameters: int = sum(p.numel() for p in model.parameters())
    num_trainable_params: int = \
        sum(param.numel() for param in model.parameters() if param.requires_grad)
    if model_name is None:
        title: str = "_Model Statistics_"
    else:
        title: str = f"_{model_name} Model Statistics_"
    print(title)
    print(
        f"Number of parameters: {num_parameters:_}\n"
        f"Number of trainable parameters: {num_trainable_params:_}"
    )
