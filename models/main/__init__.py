"""
GNN-based Suspicious Transaction Detection System

Main module for blockchain transaction classification using 
Graph Neural Networks with sequence encoding.
"""

__version__ = "0.1.0"
__author__ = "Your Team"

from . import core
from . import models
from . import baselines
from . import data
from . import training
from . import experiments

__all__ = [
    "core",
    "models",
    "baselines",
    "data",
    "training",
    "experiments",
]
