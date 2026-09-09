"""Readable reference integrals for Cartesian London orbitals.

This package favors direct formulas and independent checks over speed. It does
not import the production C++ extension.
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
from .property import (
    primitive_gradient,
    primitive_kinetic,
    primitive_magnetic_kinetic,
    primitive_moment,
    primitive_momentum,
    shell_gradient,
    shell_kinetic,
    shell_magnetic_kinetic,
    shell_moment,
    shell_momentum,
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
    "primitive_gradient",
    "primitive_kinetic",
    "primitive_magnetic_kinetic",
    "primitive_moment",
    "primitive_momentum",
    "shell_gradient",
    "shell_kinetic",
    "shell_magnetic_kinetic",
    "shell_moment",
    "shell_momentum",
    "shell_overlap",
    "ss_overlap",
]
