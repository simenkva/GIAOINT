"""Compute small finite-field one-electron matrices and first derivatives."""

from __future__ import annotations

import giao_integrals as gi
import numpy as np


def make_water_basis() -> gi.Basis:
    """Return a compact demonstration basis (not a named basis-set fit)."""

    shells = [
        gi.Shell((0.0, 0.0, 0.0), 0, [5.0, 1.0], [0.35, 0.75]),
        gi.Shell((0.0, 0.0, 0.0), 1, [1.2], [1.0]),
        gi.Shell((1.43, 1.11, 0.0), 0, [0.8], [1.0]),
        gi.Shell((-1.43, 1.11, 0.0), 0, [0.8], [1.0]),
    ]
    return gi.Basis(shells)


def main() -> None:
    basis = make_water_basis()
    nuclei = [
        gi.Nucleus(8.0, (0.0, 0.0, 0.0)),
        gi.Nucleus(1.0, (1.43, 1.11, 0.0)),
        gi.Nucleus(1.0, (-1.43, 1.11, 0.0)),
    ]
    field = gi.MagneticField((0.0, 0.0, 0.02), gauge_origin=(0.1, -0.1, 0.0))

    overlap = gi.overlap(basis, field=field)
    kinetic = gi.magnetic_kinetic(basis, field=field)
    attraction = gi.nuclear_attraction(basis, nuclei, field=field)
    shell_response, nucleus_response = gi.nuclear_attraction_nuclear_derivatives(
        basis, nuclei, field=field
    )

    assert np.allclose(overlap, overlap.conj().T)
    assert np.allclose(kinetic, kinetic.conj().T)
    assert np.allclose(attraction, attraction.conj().T)
    print(f"giao-integrals {gi.__version__}: {basis.ao_count} Cartesian AOs")
    print("S/T/V shapes:", overlap.shape, kinetic.shape, attraction.shape)
    print(
        "basis/potential response shapes:", shell_response.shape, nucleus_response.shape
    )


if __name__ == "__main__":
    main()
