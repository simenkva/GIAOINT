"""Optional independent libcint checks for mixed derivatives at B=0."""

# Loop-local closures are evaluated synchronously.
# ruff: noqa: B023

import basis_set_exchange as bse
import giao_integrals as gi
import numpy as np
from pyscf import gto


def main():
    atoms = [("O", (0.1, -0.2, 0.3)), ("H", (0.2, 0.4, 1.6)), ("H", (1.5, -0.1, -0.3))]
    basis_name = "cc-pVDZ"
    basis = {
        s: gto.basis.parse(
            bse.get_basis(basis_name, elements=[s], fmt="nwchem", header=False)
        )
        for s in ["O", "H"]
    }
    gauge = np.array([0.37, -0.29, 0.43])

    def oracle_integrals(geometry):
        oracle = gto.M(atom=geometry, basis=basis, unit="Bohr", cart=True, verbose=0)
        oracle.set_common_orig(gauge)
        overlap = oracle.intor("int1e_ovlp_cart")
        scale = 1 / np.sqrt(np.diag(overlap))

        def integral(name):
            return oracle.intor(name) * scale[:, None] * scale[None, :]

        return oracle, integral

    mol = gi.Molecule(atoms, basis_name, unit="bohr")
    oracle, integral = oracle_integrals(atoms)
    n = mol.ao_count
    offsets = mol.basis.ao_offsets
    np.testing.assert_array_equal(offsets, oracle.ao_loc_nr()[:-1])
    ao_centers = np.repeat(
        [s.center for s in mol.basis.shells],
        [s.ao_count for s in mol.basis.shells],
        axis=0,
    )
    pair_displacement = ao_centers[:, None, :] - ao_centers[None, :, :]
    # <r> origin is zero here; set_common_orig applies only to coordinate ops.
    oracle.set_common_orig((0, 0, 0))
    r = integral("int1e_r_cart")
    irp = integral("int1e_irp_cart").reshape(3, 3, n, n)
    actual = gi.overlap_mixed_derivatives(
        mol.basis, field=gi.MagneticField((0, 0, 0), gauge), threads=1
    )
    expected = np.zeros_like(actual)
    for shell, lo in enumerate(offsets):
        hi = lo + mol.basis.shells[shell].ao_count
        row = np.zeros((n, n))
        row[lo:hi, :] = 1
        column = np.zeros((n, n))
        column[:, lo:hi] = 1
        for nuclear_axis in range(3):
            # d<r_t>/dR_shell at the same fixed origin, independently from irp.
            dm = (
                -irp[:, nuclear_axis] * column
                - irp[:, nuclear_axis].transpose(0, 2, 1) * row
            )
            for field_axis in range(3):
                c = np.cross(np.eye(3)[field_axis], pair_displacement) / 2
                dc = np.cross(np.eye(3)[field_axis], np.eye(3)[nuclear_axis]) / 2
                # Centre coefficient response and ordinary Gaussian response.
                expected[shell, nuclear_axis, field_axis] = 1j * np.einsum(
                    "t,tij->ij", dc, r
                ) * (row - column) + 1j * np.einsum("ijt,tij->ij", c, dm)
    error = np.max(np.abs(actual - expected))
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    # Explicit physical operator contribution: d/dR [1/2 * (r-G) cross p].
    # Compare grouped atom derivatives to Richardson differences of libcint irp.
    field = gi.MagneticField((0, 0, 0), gauge)
    explicit = gi.magnetic_kinetic_mixed_derivatives(
        mol.basis, field=field, threads=1
    ) - gi.kinetic_mixed_derivatives(mol.basis, field=field, threads=1)

    def angular_momentum(geometry):
        _, value = oracle_integrals(geometry)
        matrix = value("int1e_irp_cart").reshape(3, 3, n, n)
        return np.array(
            [
                -0.5j
                * (
                    matrix[(axis + 1) % 3, (axis + 2) % 3]
                    - matrix[(axis + 2) % 3, (axis + 1) % 3]
                )
                for axis in range(3)
            ]
        )

    for atom in range(3):
        grouped = explicit[np.array(mol.shell_atoms) == atom].sum(axis=0)
        for nuclear_axis in range(3):

            def changed(step):
                geometry = list(atoms)
                geometry[atom] = (
                    atoms[atom][0],
                    np.array(atoms[atom][1]) + step * np.eye(3)[nuclear_axis],
                )
                return angular_momentum(geometry)

            h = 4e-4
            coarse = (changed(h) - changed(-h)) / (2 * h)
            fine = (changed(h / 2) - changed(-h / 2)) / h
            expected = (4 * fine - coarse) / 3
            error = max(error, np.max(np.abs(grouped[nuclear_axis] - expected)))
            np.testing.assert_allclose(
                grouped[nuclear_axis], expected, atol=3e-10, rtol=3e-9
            )
    print(
        f"Priority 3: H2O/{basis_name}, {n} Cartesian AOs; "
        f"max libcint/Richardson error {error:.3e}"
    )


if __name__ == "__main__":
    main()
