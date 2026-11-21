"""Optional streamed zero-field ERI centre derivatives against libcint ip1."""

import basis_set_exchange as bse
import giao_integrals as gi
import numpy as np
from pyscf import gto


def main():
    atoms = [("O", (0.1, -0.2, 0.3)), ("H", (0.2, 0.4, 1.6)), ("H", (1.5, -0.1, -0.3))]
    name = "STO-3G"
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
    offsets = mol.basis.ao_offsets
    np.testing.assert_array_equal(offsets, oracle.ao_loc_nr()[:-1])
    scale = 1 / np.sqrt(np.diag(oracle.intor("int1e_ovlp_cart")))
    ip1 = oracle.intor("int2e_ip1_cart")
    ip1 *= np.einsum("i,j,k,l->ijkl", scale, scale, scale, scale)[None, ...]
    maximum = 0.0
    count = 0
    for batch in mol.eri_derivative_batches(target_bytes=4096, threads=1):
        for row, q in enumerate(batch.quartets):
            slices = [
                slice(
                    offsets[int(s)], offsets[int(s)] + mol.basis.shells[int(s)].ao_count
                )
                for s in q
            ]
            a, b, c, d = slices
            expected = np.stack(
                [
                    -ip1[:, a, b, c, d],
                    -ip1[:, b, a, c, d].transpose(0, 2, 1, 3, 4),
                    -ip1[:, c, d, a, b].transpose(0, 3, 4, 1, 2),
                    -ip1[:, d, c, a, b].transpose(0, 3, 4, 2, 1),
                ]
            )
            actual = batch.block(row)
            maximum = max(maximum, np.max(np.abs(actual - expected)))
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
            count += 1
    print(
        f"Priority 4: H2O/{name}, {mol.ao_count} AOs; "
        f"{count} canonical quartets; max libcint error {maximum:.3e}"
    )


if __name__ == "__main__":
    main()
