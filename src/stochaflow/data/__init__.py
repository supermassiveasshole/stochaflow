"""Data package."""

from .datasets import build_cifar10_dataset, build_mnist_dataset

__all__ = [
    "build_cifar10_dataset",
    "build_mnist_dataset",
]
