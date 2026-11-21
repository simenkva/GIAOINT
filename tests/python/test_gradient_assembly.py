"""Atom assembly and complex density contractions against explicit ordered blocks."""

# Loop closures are consumed synchronously.
# ruff: noqa: B023
from itertools import product

import giao_integrals as gi
import numpy as np
import pytest
from test_moment_derivatives import FIELDS, ORIGIN, richardson

ATOMS = [("O", (0.1, -0.2, 0.3)), ("H", (0.2, 0.4, 1.6)), ("H", (1.5, -0.1, -0.3))]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize(
    "name", ["overlap", "kinetic", "magnetic_kinetic", "nuclear_attraction", "moment"]
)
def test_atom_one_electron_matches_explicit_shell_assembly(field, mixed, name):
    mol = gi.Molecule(ATOMS, "STO-3G", unit="bohr", field=field)
    kind = "mixed" if mixed else "nuclear"
    extra = (
        [(1, 1, 0)]
        if name == "moment"
        else [mol.nuclei] if name == "nuclear_attraction" else []
    )
    keywords = {"field": field}
    if mixed or name == "moment":
        keywords["threads"] = 1
    if name == "moment":
        keywords["origin"] = ORIGIN
    shell = getattr(gi, name + "_" + kind + "_derivatives")(
        mol.basis, *extra, **keywords
    )
    origin = None
    if name == "moment":
        shell, origin = shell
    elif name == "nuclear_attraction":
        shell, potential = shell
    shape = (3, 3, 3, 7, 7) if mixed else (3, 3, 7, 7)
    expected = np.zeros(shape, dtype=complex)
    np.add.at(expected, np.asarray(mol.shell_atoms), shell)
    if name == "nuclear_attraction":
        expected += potential
    args = [(1, 1, 0)] if name == "moment" else []
    kwargs = {"origin": ORIGIN} if name == "moment" else {}
    method = getattr(mol, name + "_" + kind + "_derivatives")
    result = method(*args, **kwargs, threads=1)
    if name == "moment":
        actual, actual_origin = result
        np.testing.assert_array_equal(actual_origin, origin)
    else:
        actual = result
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-14)
    out = (
        tuple(np.empty_like(x) for x in result)
        if name == "moment"
        else np.empty_like(result)
    )
    repeated = method(*args, **kwargs, out=out, threads=2 if gi.openmp_enabled() else 1)
    for a, b, c in zip(
        repeated if name == "moment" else [repeated],
        out if name == "moment" else [out],
        result if name == "moment" else [result],
        strict=True,
    ):
        assert a is b
        np.testing.assert_array_equal(a, c)
    # Independent atom-centre differences move every owned shell and potential.
    for axis in range(3):
        if mixed:

            def field_response(step):
                changed_field = gi.MagneticField(
                    np.array(field.B) + step * np.eye(3)[axis], field.gauge_origin
                )
                changed_mol = gi.Molecule(
                    ATOMS, "STO-3G", unit="bohr", field=changed_field
                )
                response = getattr(changed_mol, name + "_nuclear_derivatives")(
                    *args, **kwargs, threads=1
                )
                return response[0] if name == "moment" else response

            numerical = richardson(field_response)
            np.testing.assert_allclose(
                actual[:, :, axis], numerical, atol=8e-10, rtol=8e-9
            )
        else:

            def changed(step):
                atoms = list(ATOMS)
                atoms[0] = (ATOMS[0][0], np.array(ATOMS[0][1]) + step * np.eye(3)[axis])
                changed_mol = gi.Molecule(atoms, "STO-3G", unit="bohr", field=field)
                if name == "moment":
                    return gi.moment(
                        changed_mol.basis, *args, origin=ORIGIN, field=field
                    )
                return getattr(changed_mol, name)()

            np.testing.assert_allclose(
                actual[0, axis], richardson(changed), atol=8e-10, rtol=8e-9
            )


@pytest.mark.parametrize("field", FIELDS)
def test_moment_origin_mixed_is_separate(field):
    mol = gi.Molecule(ATOMS, "STO-3G", unit="bohr", field=field)
    atoms, origin = mol.moment_mixed_derivatives((1, 1, 0), origin=ORIGIN, threads=1)
    assert atoms.shape == (3, 3, 3, 7, 7)
    assert origin.shape == (3, 3, 7, 7)
    for axis in range(3):
        numerical = richardson(
            lambda step: gi.moment_nuclear_derivatives(
                mol.basis,
                (1, 1, 0),
                origin=ORIGIN,
                field=gi.MagneticField(
                    np.array(field.B) + step * np.eye(3)[axis], field.gauge_origin
                ),
                threads=1,
            )[1]
        )
        np.testing.assert_allclose(origin[:, axis], numerical, atol=5e-10, rtol=5e-9)


def contraction_system():
    return gi.Basis(
        [
            gi.Shell((0.1, -0.2, 0.3), 0, [0.8], [1.0]),
            gi.Shell((-0.3, 0.4, 0.2), 1, [0.6], [1.0]),
            gi.Shell((0.1, -0.2, 0.3), 0, [1.1], [1.0]),
        ]
    ), [0, 1, 0]


def explicit_contraction(basis, owners, field, jpairs, kpairs, gamma):
    result = np.zeros((2, 3), dtype=complex)
    offsets = basis.ao_offsets
    shells = basis.shells
    for q in product(range(len(shells)), repeat=4):
        s = [slice(offsets[k], offsets[k] + shells[k].ao_count) for k in q]
        a, b, c, d = s
        weight = np.zeros(tuple(shells[k].ao_count for k in q), dtype=complex)
        for p, r in jpairs:
            weight += np.einsum("nm,sl->mnls", p[b, a], r[d, c])
        for p, r in kpairs:
            weight += np.einsum("nl,sm->mnls", p[b, c], r[d, a])
        if gamma is not None:
            weight += gamma[tuple(s)]
        derivative = gi.eri_center_derivatives_shell(
            *[shells[k] for k in q], field=field
        )
        for slot in range(4):
            result[owners[q[slot]]] += np.einsum(
                "rmnls,mnls->r", derivative[slot], weight
            )
    return result


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("kind", ["j", "k", "gamma", "real_gamma", "combined"])
def test_streamed_complex_density_contraction_matches_all_ordered_quartets(field, kind):
    basis, owners = contraction_system()
    n = basis.ao_count
    rng = np.random.default_rng(281)

    def matrix():
        return rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))

    jp = (
        [(matrix(), matrix()), (matrix(), matrix())]
        if kind in ["j", "combined"]
        else []
    )
    kp = (
        [(matrix(), matrix()), (matrix(), matrix())]
        if kind in ["k", "combined"]
        else []
    )
    gamma = (
        rng.normal(size=(n,) * 4)
        if kind in ["gamma", "real_gamma", "combined"]
        else None
    )
    if kind in ["gamma", "combined"]:
        gamma = gamma + 1j * rng.normal(size=(n,) * 4)
    expected = explicit_contraction(basis, owners, field, jp, kp, gamma)
    actual = gi.eri_derivative_contraction(
        basis,
        owners,
        2,
        field=field,
        coulomb_pairs=jp,
        exchange_pairs=kp,
        gamma=gamma,
        target_bytes=1024,
        threads=1,
    )
    np.testing.assert_allclose(actual, expected, atol=4e-12, rtol=4e-12)
    out = np.empty_like(actual)
    repeated = gi.eri_derivative_contraction(
        basis,
        owners,
        2,
        field=field,
        coulomb_pairs=jp,
        exchange_pairs=kp,
        gamma=gamma,
        target_bytes=1024 * 1024,
        threads=2 if gi.openmp_enabled() else 1,
        out=out,
    )
    assert repeated is out
    np.testing.assert_array_equal(repeated, actual)
    # Derivative of a value contraction under a simultaneous owned-shell shift.
    weight = np.zeros((n,) * 4, dtype=complex)
    for p, q in jp:
        weight += np.einsum("nm,sl->mnls", p, q)
    for p, q in kp:
        weight += np.einsum("nl,sm->mnls", p, q)
    if gamma is not None:
        weight += gamma

    def value(step):
        shells = [
            gi.Shell(
                np.array([0.1, -0.2, 0.3]) + np.array([step, 0, 0]), 0, [0.8], [1.0]
            ),
            gi.Shell((-0.3, 0.4, 0.2), 1, [0.6], [1.0]),
            gi.Shell(
                np.array([0.1, -0.2, 0.3]) + np.array([step, 0, 0]), 0, [1.1], [1.0]
            ),
        ]
        return np.sum(
            gi.eri(
                gi.Basis(shells),
                field=field,
                storage="full",
                max_bytes=n**4 * 16,
                threads=1,
            )
            * weight
        )

    np.testing.assert_allclose(actual[0, 0], richardson(value), atol=2e-9, rtol=2e-9)


def test_molecule_contraction_and_bad_inputs():
    mol = gi.Molecule([("H", (0, 0, 0)), ("H", (0, 0, 1.4))], "STO-3G", unit="bohr")
    density = np.eye(2)
    expected = gi.eri_derivative_contraction(
        mol.basis, mol.shell_atoms, 2, coulomb_pairs=[(density, density)], threads=1
    )
    np.testing.assert_array_equal(
        mol.eri_derivative_contraction(coulomb_pairs=[(density, density)], threads=1),
        expected,
    )
    for kwargs in [
        {},
        {"gamma": np.empty((2,) * 4, dtype=np.float32)},
        {"gamma": np.empty((2,) * 4).transpose(1, 0, 2, 3)},
        {"coulomb_pairs": [(np.eye(3), np.eye(3))]},
    ]:
        with pytest.raises(ValueError):
            mol.eri_derivative_contraction(**kwargs)
    for out in [
        np.empty((2, 3)),
        np.empty((3, 2), dtype=complex),
        np.empty((2, 3), dtype=complex)[:, ::-1],
    ]:
        with pytest.raises(ValueError):
            mol.eri_derivative_contraction(coulomb_pairs=[(density, density)], out=out)


def test_canonical_orbits_permute_ao_axes_and_centre_slots():
    basis = gi.Basis(
        [
            gi.Shell((0.1, -0.2, 0.3), 0, [0.8], [1.0]),
            gi.Shell((-0.3, 0.4, 0.2), 1, [0.6], [1.0]),
        ]
    )
    owners = [0, 1]
    n = basis.ao_count
    field = FIELDS[1]
    expanded = np.zeros((2, 3, n, n, n, n), dtype=complex)
    ordered = np.zeros_like(expanded)
    permutations = [(0, 1, 2, 3), (2, 3, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0)]

    def slices(q):
        return tuple(
            slice(basis.ao_offsets[k], basis.ao_offsets[k] + basis.shells[k].ao_count)
            for k in q
        )

    for batch in gi.eri_derivative_batches(basis, field=field, threads=1):
        for row, indices in enumerate(batch.quartets):
            q = tuple(map(int, indices))
            block = batch.block(row)
            seen = set()
            for orbit, perm in enumerate(permutations):
                image = tuple(q[k] for k in perm)
                if image in seen:
                    continue
                seen.add(image)
                changed = block[list(perm)].transpose(0, 1, *(2 + k for k in perm))
                if orbit >= 2:
                    changed = changed.conj()
                for slot in range(4):
                    expanded[
                        (owners[image[slot]], slice(None), *slices(image))
                    ] += changed[slot]
    for q in product(range(2), repeat=4):
        block = gi.eri_center_derivatives_shell(
            *[basis.shells[k] for k in q], field=field
        )
        for slot in range(4):
            ordered[(owners[q[slot]], slice(None), *slices(q))] += block[slot]
    np.testing.assert_allclose(expanded, ordered, atol=3e-14, rtol=3e-14)


def test_dense_gamma_is_read_without_copy_and_out_cannot_alias(monkeypatch):
    import giao_integrals.derivative_assembly as assembly

    basis = gi.Basis([gi.Shell((0, 0, 0), 0, [0.8], [1.0])])
    gamma = np.ones((1, 1, 1, 1), dtype=np.float64)
    gamma.flags.writeable = False
    native = assembly._contract_eri_derivative_batch
    seen = []

    def checked(*args, **kwargs):
        assert args[8] is gamma
        seen.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(assembly, "_contract_eri_derivative_batch", checked)
    gi.eri_derivative_contraction(basis, [0], 1, gamma=gamma, threads=1)
    assert seen
    buffer = np.ones(3, dtype=complex)
    with pytest.raises(ValueError, match="overlap"):
        gi.eri_derivative_contraction(
            basis,
            [0],
            1,
            gamma=buffer[:1].reshape(1, 1, 1, 1),
            out=buffer.reshape(1, 3),
        )


def test_atom_moment_output_validation():
    mol = gi.Molecule([("H", (0, 0, 0))], "STO-3G", unit="bohr")
    with pytest.raises(ValueError):
        mol.moment_nuclear_derivatives((1, 0, 0), out=np.empty((1, 3, 1, 1), complex))
    with pytest.raises(ValueError):
        mol.overlap_nuclear_derivatives(out=np.empty((1, 3, 1, 1)))
    with pytest.raises(ValueError):
        mol.moment_mixed_derivatives((1, 0, 0), threads=0)


def test_contraction_screening_is_opt_in_and_uses_derivative_batches(monkeypatch):
    import giao_integrals.derivative_assembly as assembly

    basis = gi.Basis(
        [gi.Shell((0, 0, 0), 0, [0.8], [1.0]), gi.Shell((30, 0, 0), 1, [0.9], [1.0])]
    )
    density = np.eye(4, dtype=complex)
    native = assembly.eri_derivative_batches
    counts = []

    def checked(*args, **kwargs):
        assert kwargs["screening_threshold"] == 1e-8
        for batch in native(*args, **kwargs):
            counts.append(batch.screened_count)
            yield batch

    reference = gi.eri_derivative_contraction(
        basis, [0, 1], 2, coulomb_pairs=[(density, density)], threads=1
    )
    monkeypatch.setattr(assembly, "eri_derivative_batches", checked)
    result = gi.eri_derivative_contraction(
        basis,
        [0, 1],
        2,
        coulomb_pairs=[(density, density)],
        screening_threshold=1e-8,
        threads=1,
    )
    assert sum(counts) > 0
    np.testing.assert_allclose(result, reference, atol=1e-12, rtol=1e-12)
