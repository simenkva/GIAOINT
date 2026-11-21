"""TD4.0 zero-phase-field comparison with libcint coordinate/angular operators."""

import basis_set_exchange as bse
import giao_integrals as gi
import numpy as np
from pyscf import gto


def main():
    atoms = [("O", (0.1, -0.2, 0.3)), ("H", (0.2, 0.4, 1.6)), ("H", (1.5, -0.1, -0.3))]
    name = "cc-pVDZ"
    origin = (0.37, -0.29, 0.43)
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
    oracle.set_common_orig(origin)
    scale = 1 / np.sqrt(np.diag(oracle.intor("int1e_ovlp_cart")))

    def integral(op):
        return oracle.intor(op) * scale[:, None] * scale[None, :]

    # libcint returns i*(r-G) cross p = (r-G) cross gradient as real values.
    angular = -1j * integral("int1e_cg_irxp_cart")
    maximum = 0.0
    for axis, component in enumerate("xyz"):
        actual = gi.kinetic_angular_momentum(
            mol.basis, component, origin=origin, threads=1
        )
        maximum = max(maximum, np.max(np.abs(actual - angular[axis])))
        np.testing.assert_allclose(actual, angular[axis], atol=3e-12, rtol=3e-12)
    n = mol.ao_count
    q = integral("int1e_rr_cart").reshape(3, 3, n, n)
    for i in range(3):
        for j in range(3):
            powers = tuple(int(k == i) + int(k == j) for k in range(3))
            actual = gi.moment(mol.basis, powers, origin=origin)
            maximum = max(maximum, np.max(np.abs(actual - q[i, j])))
            np.testing.assert_allclose(actual, q[i, j], atol=3e-12, rtol=3e-12)
    c = np.array([0.12, -0.08, 0.03])
    expected = (
        integral("int1e_kin_cart")
        + 0.5 * np.einsum("i,imn->mn", c, angular)
        + 0.125
        * (
            np.dot(c, c) * np.trace(q, axis1=0, axis2=1)
            - np.einsum("i,j,ijmn->mn", c, c, q)
        )
    )
    actual = gi.magnetic_kinetic_operator(
        mol.basis, operator_field=c, origin=origin, threads=1
    )
    maximum = max(maximum, np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    print(f"TD4: H2O/{name}, {n} Cartesian AOs; max libcint disagreement {maximum:.3e}")


if __name__ == "__main__":
    main()
