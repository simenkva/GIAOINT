"""Load a real basis set from the Basis Set Exchange and cross-check
zero-field overlap, kinetic, nuclear-attraction, and ERI matrices against
PySCF/libcint.

This file is deliberately standalone, like ``tests/external/pyscf_zero_field.py``.
It requires the optional ``basis_set_exchange`` and ``pyscf`` packages, which
are not part of this project's dependencies:

    pip install basis_set_exchange pyscf
"""

from __future__ import annotations

import time
from collections.abc import Callable

import basis_set_exchange as bse
import giao_integrals as gi
import numpy as np
import pyscf
from pyscf import gto

BASIS_NAME = "cc-pVDZ"

ATOMS = [
    ("O", (0.0, 0.0, 0.21747)),
    ("H", (0.0, 1.44488, -0.86989)),
    ("H", (0.0, -1.44488, -0.86989)),
]


def _element_shells(basis_name: str, symbol: str) -> list[dict]:
    z = bse.lut.element_Z_from_sym(symbol)
    data = bse.get_basis(basis_name, elements=[symbol])
    return data["elements"][str(z)]["electron_shells"]


def load_bse_basis(
    basis_name: str, atoms: list[tuple[str, tuple[float, float, float]]]
) -> gi.Basis:
    """Build a Cartesian giao_integrals.Basis from a named BSE basis set."""

    shell_cache: dict[str, list[dict]] = {}
    shells: list[gi.Shell] = []
    for symbol, center in atoms:
        if symbol not in shell_cache:
            shell_cache[symbol] = _element_shells(basis_name, symbol)
        specs: list[tuple[int, list[float], list]] = []
        for entry in shell_cache[symbol]:
            exponents = [float(x) for x in entry["exponents"]]
            angular_momenta = entry["angular_momentum"]
            coefficient_rows = entry["coefficients"]
            if len(angular_momenta) == 1:
                # A general-contraction shell: every row belongs to this L.
                coefficients = [[float(c) for c in row] for row in coefficient_rows]
                specs.append((angular_momenta[0], exponents, coefficients))
            else:
                # A shared-exponent shell (e.g. Pople "SP"): one row per L.
                for momentum, row in zip(
                    angular_momenta, coefficient_rows, strict=True
                ):
                    specs.append((momentum, exponents, [float(c) for c in row]))
        # PySCF (like most NWChem-format consumers) places every shell of a
        # given angular momentum together, in increasing L order, even when
        # BSE interleaves them inside shared-exponent "SP" blocks. Sorting is
        # stable, so shells within one L keep their original relative order.
        for momentum, exponents, coefficients in sorted(specs, key=lambda s: s[0]):
            shells.append(gi.Shell(center, momentum, exponents, coefficients))
    return gi.Basis(shells)


def build_pyscf_molecule(
    basis_name: str, atoms: list[tuple[str, tuple[float, float, float]]]
) -> gto.Mole:
    """Build the matching Cartesian PySCF molecule from the same BSE data."""

    elements = sorted({symbol for symbol, _ in atoms})
    mol = gto.Mole()
    mol.atom = list(atoms)
    mol.unit = "Bohr"
    mol.cart = True
    mol.basis = {
        symbol: gto.basis.parse(
            bse.get_basis(basis_name, elements=[symbol], fmt="nwchem", header=False)
        )
        for symbol in elements
    }
    mol.build(verbose=0)
    return mol


def _timed(label: str, build: Callable[[], np.ndarray]) -> np.ndarray:
    start = time.perf_counter()
    result = build()
    elapsed = time.perf_counter() - start
    print(f"{label}: built in {elapsed * 1e3:.3f} ms")
    return result


def main() -> None:
    basis = load_bse_basis(BASIS_NAME, ATOMS)
    mol = build_pyscf_molecule(BASIS_NAME, ATOMS)
    assert basis.ao_count == mol.nao

    nuclei = [
        gi.Nucleus(float(mol.atom_charge(i)), tuple(mol.atom_coord(i)))
        for i in range(mol.natm)
    ]

    print(f"giao-integrals {gi.__version__} vs PySCF {pyscf.__version__}")
    print(f"basis: {BASIS_NAME}, {basis.ao_count} Cartesian AOs")

    # PySCF normalizes shared Cartesian components by one radial factor per
    # shell, while giao_integrals normalizes every Cartesian component
    # individually. Rescale PySCF's matrices to the same per-AO convention
    # before comparing (see tests/external/pyscf_zero_field.py).
    raw_overlap = _timed("pyscf overlap", lambda: mol.intor("int1e_ovlp_cart"))
    scale = 1.0 / np.sqrt(np.diag(raw_overlap))

    def rescale(matrix: np.ndarray) -> np.ndarray:
        return scale[:, None] * matrix * scale[None, :]

    comparisons = {
        "overlap": (
            _timed("giao overlap", lambda: gi.overlap(basis)),
            rescale(raw_overlap),
        ),
        "kinetic": (
            _timed("giao kinetic", lambda: gi.kinetic(basis)),
            rescale(_timed("pyscf kinetic", lambda: mol.intor("int1e_kin_cart"))),
        ),
        "nuclear attraction": (
            _timed(
                "giao nuclear attraction",
                lambda: gi.nuclear_attraction(basis, nuclei),
            ),
            rescale(
                _timed("pyscf nuclear attraction", lambda: mol.intor("int1e_nuc_cart"))
            ),
        ),
    }

    for name, (actual, expected) in comparisons.items():
        error = np.max(np.abs(actual.real - expected))
        imag = np.max(np.abs(actual.imag))
        print(f"{name}: max real error {error:.3e}, max imaginary magnitude {imag:.3e}")

    raw_eri = _timed("pyscf ERI", lambda: mol.intor("int2e_cart"))
    expected_eri = np.einsum("i,j,k,l,ijkl->ijkl", scale, scale, scale, scale, raw_eri)
    actual_eri = _timed(
        "giao ERI",
        lambda: gi.eri(basis, storage="full", max_bytes=64 * 1024**2),
    )
    eri_error = np.max(np.abs(actual_eri.real - expected_eri))
    eri_imag = np.max(np.abs(actual_eri.imag))
    print(
        f"ERI: max real error {eri_error:.3e}, max imaginary magnitude {eri_imag:.3e}"
    )


if __name__ == "__main__":
    main()
