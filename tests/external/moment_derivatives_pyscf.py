"""Optional zero-field libcint oracle for literal moment centre derivatives.

Run with giao-integrals[molecule] and PySCF installed. Geometries are in bohr;
Cartesian AO normalization is matched explicitly. int1e_irp differentiates
ket *spatial* coordinates, hence the minus sign for ket-centre derivatives.
"""

import basis_set_exchange as bse
import giao_integrals as gi
import numpy as np
from pyscf import gto


def main():
    atoms = [("O", (0.1, -0.2, 0.3)), ("H", (0.2, 0.4, 1.6)), ("H", (1.5, -0.1, -0.3))]
    name = "cc-pVDZ"
    mol = gi.Molecule(atoms, name, unit="bohr")
    oracle = gto.M(
        atom=atoms,
        unit="Bohr",
        cart=True,
        verbose=0,
        basis={
            s: gto.basis.parse(
                bse.get_basis(name, elements=[s], fmt="nwchem", header=False)
            )
            for s in ["O", "H"]
        },
    )
    offsets = np.asarray(mol.basis.ao_offsets)
    np.testing.assert_array_equal(offsets, oracle.ao_loc_nr()[:-1])
    scale = 1 / np.sqrt(np.diag(oracle.intor("int1e_ovlp_cart")))
    origin = (0.37, -0.29, 0.43)
    oracle.set_common_orig(origin)
    irp = oracle.intor("int1e_irp_cart").reshape(3, 3, mol.ao_count, mol.ao_count)
    irp *= scale[None, None, :, None] * scale[None, None, None, :]
    maximum = 0.0
    for component in range(3):
        powers = [int(k == component) for k in range(3)]
        actual, d_origin = gi.moment_nuclear_derivatives(
            mol.basis, powers, origin=origin, threads=1
        )
        expected = np.zeros_like(actual)
        for shell, lo in enumerate(offsets):
            hi = lo + mol.basis.shells[shell].ao_count
            for axis in range(3):
                expected[shell, axis, :, lo:hi] -= irp[component, axis, :, lo:hi]
                expected[shell, axis, lo:hi, :] -= irp[component, axis, :, lo:hi].T
        error = np.max(np.abs(actual - expected))
        maximum = max(maximum, error)
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
        origin_expected = np.zeros_like(d_origin)
        origin_expected[component] = -gi.overlap(mol.basis)
        np.testing.assert_allclose(d_origin, origin_expected, atol=3e-14)
    print(
        f"Priority 1: H2O/{name}, {mol.ao_count} Cartesian AOs; "
        f"max libcint moment-centre error {maximum:.3e}"
    )


if __name__ == "__main__":
    main()
