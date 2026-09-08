"""Readable reference integrals for Cartesian London orbitals.

This package favors direct formulas and independent checks over speed. It does
not import the future C++ extension.
"""

from .gaussian import (
    AngularMomentum,
    MagneticField,
    PrimitiveGaussian,
    Shell,
    Vector3,
    cartesian_components,
    double_factorial,
    gaussian_value,
    primitive_normalization,
)
from .overlap import (
    GaussianPair,
    contracted_component_overlap,
    contraction_normalization,
    gaussian_product,
    primitive_overlap,
    shell_overlap,
    ss_overlap,
)

__all__ = [
    "AngularMomentum",
    "GaussianPair",
    "MagneticField",
    "PrimitiveGaussian",
    "Shell",
    "Vector3",
    "cartesian_components",
    "contracted_component_overlap",
    "contraction_normalization",
    "double_factorial",
    "gaussian_product",
    "gaussian_value",
    "primitive_normalization",
    "primitive_overlap",
    "shell_overlap",
    "ss_overlap",
]

__all__ = [
    "AngularMomentum",
    "GaussianPair",
    "MagneticField",
    "PrimitiveGaussian",
    "Shell",
    "Vector3",
    "cartesian_components",
    "contracted_component_overlap",
    "double_factorial",
    "gaussian_product",
    "gaussian_value",
    "primitive_normalization",
    "primitive_overlap",
    "shell_overlap",
    "ss_overlap",
]
