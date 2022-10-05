from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch import Tensor


def accuracy_loss(pred: "Tensor", target: "Tensor") -> "Tensor":
    """Acuracy to multi-target loss.
    Reduce the accuracy loss is mean
    Args:
        pred (Tensor): probability of the target.
        target (Tensor): binary target.
    """
    return 1 - (pred * target).sum(1).mean()
