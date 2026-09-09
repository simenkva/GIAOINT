"""Optional zero-field comparison against PySCF/libcint.

Run from an environment containing PySCF and an installed ``giao_integrals``.
This file is deliberately standalone because PySCF is not a test dependency.
"""

from __future__ import annotations

import giao_integrals as gi
import numpy as np
from pyscf import gto


def main() -> None:
    center_h = (-0.3, 0.2, 0.5)
    center_he = (0.6, -0.4, 0.1)
    center_li = (0.2, 0.7, -0.6)
    exponents_h = (2.4, 0.7)
    coefficients_h = (0.2, 0.8)
    exponents_he = (1.9, 0.44)
    coefficients_he = (-0.22, 0.93)

    molecule = gto.Mole()
    molecule.atom = [("H", center_h), ("He", center_he), ("Li", center_li)]
    molecule.unit = "Bohr"
    molecule.cart = True
    molecule.spin = 0
    molecule.basis = {
        "H": [
            [
                0,
                [exponents_h[0], coefficients_h[0]],
                [exponents_h[1], coefficients_h[1]],
            ]
        ],
        "He": [
            [
                1,
                [exponents_he[0], coefficients_he[0]],
                [exponents_he[1], coefficients_he[1]],
            ]
        ],
        "Li": [[2, [1.1, 1.0]]],
    }
    molecule.build(verbose=0)

    basis = gi.Basis(
        [
            gi.Shell(center_h, 0, exponents_h, coefficients_h),
            gi.Shell(center_he, 1, exponents_he, coefficients_he),
            gi.Shell(center_li, 2, (1.1,), (1.0,)),
        ]
    )
    nuclei = [
        gi.Nucleus(1.0, center_h),
        gi.Nucleus(2.0, center_he),
        gi.Nucleus(3.0, center_li),
    ]

    raw_overlap = molecule.intor("int1e_ovlp_cart")
    raw_attraction = molecule.intor("int1e_nuc_cart")
    raw_eri = molecule.intor("int2e_cart")
    # PySCF's Cartesian components share a radial shell normalization, whereas
    # giao_integrals normalizes every Cartesian component individually.
    # Convert the external matrices explicitly to the public convention.
    cartesian_scale = 1.0 / np.sqrt(np.diag(raw_overlap))
    expected_overlap = cartesian_scale[:, None] * raw_overlap * cartesian_scale[None, :]
    expected_attraction = (
        cartesian_scale[:, None] * raw_attraction * cartesian_scale[None, :]
    )
    expected_eri = np.einsum(
        "i,j,k,l,ijkl->ijkl",
        cartesian_scale,
        cartesian_scale,
        cartesian_scale,
        cartesian_scale,
        raw_eri,
    )
    actual_overlap = gi.overlap(basis)
    actual_attraction = gi.nuclear_attraction(basis, nuclei)
    actual_eri = gi.eri(basis, storage="full", max_bytes=200_000)

    np.testing.assert_allclose(
        actual_overlap.real, expected_overlap, atol=2.0e-12, rtol=2.0e-11
    )
    np.testing.assert_allclose(actual_overlap.imag, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        actual_attraction.real,
        expected_attraction,
        atol=2.0e-12,
        rtol=2.0e-11,
    )
    np.testing.assert_allclose(actual_attraction.imag, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        actual_eri.real, expected_eri, atol=2.0e-12, rtol=2.0e-11
    )
    np.testing.assert_allclose(actual_eri.imag, 0.0, atol=2.0e-15)

    overlap_error = np.max(np.abs(actual_overlap.real - expected_overlap))
    attraction_error = np.max(np.abs(actual_attraction.real - expected_attraction))
    eri_error = np.max(np.abs(actual_eri.real - expected_eri))
    print(f"PySCF {__import__('pyscf').__version__}")
    print(f"maximum overlap error: {overlap_error:.3e}")
    print(f"maximum nuclear-attraction error: {attraction_error:.3e}")
    print(f"maximum ERI error: {eri_error:.3e}")


if __name__ == "__main__":
    main()
