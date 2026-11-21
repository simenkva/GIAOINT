"""Optional libcint oracle: property centre and B responses at zero field."""

import basis_set_exchange as bse
import giao_integrals as gi
import numpy as np
from pyscf import gto


def main():
    atoms = [("O", (0.1, -0.2, 0.3)), ("H", (0.2, 0.4, 1.6)), ("H", (1.5, -0.1, -0.3))]
    basis_name = "cc-pVDZ"
    mol = gi.Molecule(atoms, basis_name, unit="bohr")
    oracle = gto.M(
        atom=atoms,
        unit="Bohr",
        cart=True,
        verbose=0,
        basis={
            s: gto.basis.parse(
                bse.get_basis(basis_name, elements=[s], fmt="nwchem", header=False)
            )
            for s in ["O", "H"]
        },
    )
    scale = 1 / np.sqrt(np.diag(oracle.intor("int1e_ovlp_cart")))

    def integral(name):
        return oracle.intor(name) * scale[..., None] * scale[None, ...]

    n = mol.ao_count
    overlap = integral("int1e_ovlp_cart")
    irp = integral("int1e_irp_cart").reshape(3, 3, n, n)
    hessian = integral("int1e_ipovlpip_cart").reshape(3, 3, n, n)
    rr = integral("int1e_rr_cart").reshape(3, 3, n, n)
    offsets = mol.basis.ao_offsets
    np.testing.assert_array_equal(offsets, oracle.ao_loc_nr()[:-1])
    ao_centers = np.repeat(
        [s.center for s in mol.basis.shells],
        [s.ao_count for s in mol.basis.shells],
        axis=0,
    )
    gauge = np.array([0.37, -0.29, 0.43])
    field = gi.MagneticField((0, 0, 0), gauge)
    maximum = 0.0
    for component, name in enumerate(["x", "y", "z"]):
        expected = np.zeros((len(offsets), 3, n, n))
        for shell, lo in enumerate(offsets):
            hi = lo + mol.basis.shells[shell].ao_count
            for axis in range(3):
                expected[shell, axis, lo:hi, :] -= hessian[axis, component, lo:hi, :]
                expected[shell, axis, :, lo:hi] += hessian[axis, component, :, lo:hi]
        actual = gi.gradient_nuclear_derivatives(mol.basis, name, threads=1)
        maximum = max(maximum, np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
        np.testing.assert_allclose(
            gi.momentum_nuclear_derivatives(mol.basis, name, threads=1),
            -1j * expected,
            atol=3e-12,
            rtol=3e-12,
        )
        for axis in range(3):
            cross_pair = (
                np.cross(
                    np.eye(3)[axis], ao_centers[:, None, :] - ao_centers[None, :, :]
                )
                / 2
            )
            cross_ket = np.cross(np.eye(3)[axis], ao_centers - gauge) / 2
            pair_phase = 1j * np.einsum("ijt,tij->ij", cross_pair, irp[:, component])
            # Spatial differentiation of the ket's B-dependent London phase.
            ket_phase = -1j * cross_ket[None, :, component] * overlap
            expected_magnetic = pair_phase + ket_phase
            actual = gi.gradient_magnetic_derivatives(
                mol.basis, name, field=field, threads=1
            )[axis]
            maximum = max(maximum, np.max(np.abs(actual - expected_magnetic)))
            np.testing.assert_allclose(
                actual, expected_magnetic, atol=3e-12, rtol=3e-12
            )
            np.testing.assert_allclose(
                gi.momentum_magnetic_derivatives(
                    mol.basis, name, field=field, threads=1
                )[axis],
                -1j * expected_magnetic,
                atol=3e-12,
                rtol=3e-12,
            )
            powers = [int(k == component) for k in range(3)]
            expected_moment = 1j * np.einsum(
                "ijt,tij->ij", cross_pair, rr[:, component]
            )
            actual = gi.moment_magnetic_derivatives(
                mol.basis, powers, field=field, threads=1
            )[axis]
            maximum = max(maximum, np.max(np.abs(actual - expected_moment)))
            np.testing.assert_allclose(actual, expected_moment, atol=3e-12, rtol=3e-12)
    print(
        f"Priority 2: H2O/{basis_name}, {n} Cartesian AOs; "
        f"max libcint error {maximum:.3e}"
    )


if __name__ == "__main__":
    main()
