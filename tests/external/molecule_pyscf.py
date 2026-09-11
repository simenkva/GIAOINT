"""Cross-check the molecule wrapper against PySCF using BSE's basis data.

Run with an installed giao-integrals[molecule] and the optional PySCF package.
Geometry is supplied to PySCF in bohr after the wrapper's documented unit
conversion, so differing physical-constant versions do not affect the test.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import basis_set_exchange as bse
import giao_integrals as gi
import numpy as np
from pyscf import gto

XYZ = "3\nwater in angstrom\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24\n"


def check(name: str, source, basis_format=None) -> None:
    molecule = gi.Molecule.from_xyz(XYZ, source, basis_format=basis_format)
    oracle = gto.M(
        atom=molecule.atoms,
        basis={
            symbol: gto.basis.parse(
                bse.get_basis(name, elements=[symbol], fmt="nwchem", header=False)
            )
            for symbol in ("H", "O")
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    raw_overlap = oracle.intor("int1e_ovlp_cart")
    scale = 1.0 / np.sqrt(np.diag(raw_overlap))
    actual = molecule.integrals(eri=True)
    for value, operator in [
        (actual.overlap, "int1e_ovlp_cart"),
        (actual.kinetic, "int1e_kin_cart"),
        (actual.nuclear_attraction, "int1e_nuc_cart"),
    ]:
        expected = np.einsum("i,j,ij->ij", scale, scale, oracle.intor(operator))
        np.testing.assert_allclose(value, expected, atol=2e-11, rtol=2e-11)
    expected_eri = np.einsum(
        "i,j,k,l,ijkl->ijkl", scale, scale, scale, scale, oracle.intor("int2e_cart")
    )
    np.testing.assert_allclose(actual.eri, expected_eri, atol=2e-11, rtol=2e-11)
    print(
        f"{name} {basis_format or 'named'}: {molecule.ao_count} AOs; "
        f"max ERI error {np.max(np.abs(actual.eri - expected_eri)):.3e}"
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        for name in ("STO-3G", "6-31G", "cc-pVDZ"):
            check(name, name)
        for fmt, suffix in [("gaussian94", "gbs"), ("nwchem", "nw"), ("json", "json")]:
            path = Path(directory) / f"basis.{suffix}"
            path.write_text(bse.get_basis("cc-pVDZ", elements=[1, 8], fmt=fmt))
            check("cc-pVDZ", path, basis_format=fmt)


if __name__ == "__main__":
    main()
