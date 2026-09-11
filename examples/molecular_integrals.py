"""Compute integrals from an XYZ file and a named or file-based basis.

Install once with ``pip install '.[molecule]'``, then run this example.
"""

from pathlib import Path

import giao_integrals as gi


def main() -> None:
    molecule = gi.Molecule.from_xyz(
        Path(__file__).with_name("water.xyz"), basis="STO-3G"
    )
    integrals = molecule.integrals(eri=True)
    print(molecule)
    print("S/T/V/Hcore:", integrals.overlap.shape, integrals.core_hamiltonian.shape)
    print("ERIs:", integrals.eri.shape)


if __name__ == "__main__":
    main()
