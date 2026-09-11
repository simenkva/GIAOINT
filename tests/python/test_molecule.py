from __future__ import annotations

import builtins
import copy
import json

import giao_integrals as gi
import numpy as np
import pytest
from giao_integrals.molecule import BOHR_IN_ANGSTROM

XYZ = "2\nHydrogen\nH 0 0 0\nH 0 0 0.74\n"
HYDROGEN = {
    "elements": {
        "1": {
            "electron_shells": [
                {
                    "function_type": "gto",
                    "angular_momentum": [0],
                    "exponents": ["1.2D+00", "0.35"],
                    "coefficients": [["0.3", "0.8"]],
                }
            ]
        }
    }
}


def test_xyz_files_units_and_atom_metadata(tmp_path):
    path = tmp_path / "hydrogen.xyz"
    path.write_text("\ufeff2\n\nh 0 0 0\n1 0 0 0.74\n\n", encoding="utf8")
    mol = gi.Molecule.from_xyz(path, basis=HYDROGEN)
    assert mol.atoms[0] == ("H", (0.0, 0.0, 0.0))
    assert mol.atoms[1][1][2] == pytest.approx(0.74 / BOHR_IN_ANGSTROM)
    assert mol.shell_atoms == (0, 1)
    assert mol.ao_count == 2
    assert mol.nuclei[1].charge == 1
    assert mol.nuclei[1].center == mol.atoms[1][1]
    assert "cartesian_aos=2" in repr(mol)
    bohr = gi.Molecule(mol.atoms, HYDROGEN, unit="bohr")
    np.testing.assert_array_equal(bohr.overlap(), mol.overlap())
    inline = gi.Molecule.from_xyz(XYZ, HYDROGEN)
    np.testing.assert_array_equal(inline.overlap(), mol.overlap())
    with pytest.raises(AttributeError):
        mol.basis = gi.Basis([])


@pytest.mark.parametrize(
    "field", [None, gi.MagneticField((0.13, -0.21, 0.17), (0.1, 0.2, -0.3))]
)
def test_molecular_integrals_match_explicit_low_level_construction(field):
    # Construct the expected basis/nuclei independently, including conversion.
    z = 0.74 / BOHR_IN_ANGSTROM
    basis = gi.Basis(
        [
            gi.Shell((0, 0, 0), 0, [1.2, 0.35], [0.3, 0.8]),
            gi.Shell((0, 0, z), 0, [1.2, 0.35], [0.3, 0.8]),
        ]
    )
    nuclei = [gi.Nucleus(1, (0, 0, 0)), gi.Nucleus(1, (0, 0, z))]
    mol = gi.Molecule.from_xyz(XYZ, HYDROGEN, field=field)
    integrals = mol.integrals(eri=True)
    assert isinstance(integrals, gi.MolecularIntegrals)
    expected_s = gi.overlap(basis, field=field)
    expected_t = gi.kinetic(basis, field=field)
    expected_v = gi.nuclear_attraction(basis, nuclei, field=field)
    expected_physical = gi.magnetic_kinetic(basis, field=field)
    expected_eri = gi.eri(basis, field=field, storage="full", max_bytes=1024)
    for actual, expected in [
        (integrals.overlap, expected_s),
        (integrals.kinetic, expected_t),
        (integrals.nuclear_attraction, expected_v),
        (integrals.core_hamiltonian, expected_physical + expected_v),
        (mol.core_hamiltonian(), expected_physical + expected_v),
        (mol.magnetic_kinetic(), expected_physical),
        (integrals.eri, expected_eri),
        (mol.eri(), expected_eri),
    ]:
        assert actual.dtype == np.complex128
        assert actual.flags.c_contiguous
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    assert mol.integrals().eri is None
    if field is not None:
        assert np.max(np.abs(expected_physical - expected_t)) > 1e-5
    batches = list(mol.eri_batches(target_bytes=256))
    expected_batches = list(gi.eri_batches(basis, field=field, target_bytes=256))
    assert len(batches) == len(expected_batches)
    for batch, expected in zip(batches, expected_batches, strict=True):
        np.testing.assert_array_equal(batch.quartets, expected.quartets)
        np.testing.assert_allclose(batch.values, expected.values, atol=2e-14)
    with pytest.raises(MemoryError, match="requires"):
        mol.eri(max_bytes=0)
    with pytest.raises(MemoryError, match="requires"):
        mol.integrals(eri=True, max_eri_bytes=0)
    with pytest.raises(ValueError, match="screening_threshold"):
        mol.eri(screening_threshold=-1)


def test_general_and_combined_sp_contractions_and_stable_shell_order():
    data = copy.deepcopy(HYDROGEN)
    data["elements"]["1"]["electron_shells"] = [
        {"angular_momentum": [2], "exponents": [0.8], "coefficients": [[1.0]]},
        {
            "angular_momentum": [0, 1],
            "exponents": [1.4, 0.4],
            "coefficients": [[0.3, 0.8], [0.6, -0.2]],
        },
        {
            "angular_momentum": [0],
            "exponents": [1.5, 0.5],
            "coefficients": [[0.4, 0.7], [0.8, -0.3]],
        },
    ]
    mol = gi.Molecule([("H", (0, 0, 0))], data)
    expected = gi.Basis(
        [
            gi.Shell((0, 0, 0), 0, [1.4, 0.4], [0.3, 0.8]),
            gi.Shell((0, 0, 0), 0, [1.5, 0.5], [[0.4, 0.7], [0.8, -0.3]]),
            gi.Shell((0, 0, 0), 1, [1.4, 0.4], [0.6, -0.2]),
            gi.Shell((0, 0, 0), 2, [0.8], [1.0]),
        ]
    )
    assert mol.ao_count == 12
    assert mol.shell_atoms == (0, 0, 0, 0)
    np.testing.assert_allclose(mol.overlap(), gi.overlap(expected), atol=2e-14)


@pytest.mark.parametrize("kind", ["dict", "text", "file"])
def test_json_does_not_require_optional_dependency(kind, tmp_path, monkeypatch):
    original_import = builtins.__import__

    def without_bse(name, *args, **kwargs):
        if name == "basis_set_exchange":
            raise ImportError("deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_bse)
    source = HYDROGEN
    if kind == "text":
        source = json.dumps(HYDROGEN)
    elif kind == "file":
        source = tmp_path / "basis.json"
        source.write_text(json.dumps(HYDROGEN))
    assert gi.Molecule.from_xyz(XYZ, source).ao_count == 2
    with pytest.raises(ImportError, match=r"giao-integrals\[molecule\]"):
        gi.Molecule.from_xyz(XYZ, "STO-3G")


@pytest.mark.parametrize(
    "text, message",
    [
        ("x\ncomment\nH 0 0 0", "atom count"),
        ("0\ncomment\n", "positive"),
        ("2\ncomment\nH 0 0 0", "comment line and 2"),
        ("1\nH 0 0 0\n", "comment line"),
        ("1\ncomment\nH 0 0", "three coordinates"),
        ("1\ncomment\nH 0 0 nan", "finite"),
        ("1\ncomment\nH 0 0 inf", "finite"),
        ("1\ncomment\nXx 0 0 0", "Unknown element"),
        ("1\ncomment\nH 0 0 0\nH 1 0 0", "one XYZ frame"),
        (XYZ + XYZ, "one XYZ frame"),
    ],
)
def test_invalid_xyz_is_diagnostic(text, message):
    with pytest.raises(ValueError, match=message):
        gi.Molecule.from_xyz(text, HYDROGEN)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"ecp_electrons": 2}, "ECP"),
        ({"electron_shells": []}, "no electron shells"),
        ({"electron_shells": [{"function_type": "sto"}]}, "Gaussian"),
        ({"electron_shells": [{"angular_momentum": [-1]}]}, "non-negative"),
        ({"electron_shells": [{"angular_momentum": [0.5]}]}, "non-negative"),
        (
            {
                "electron_shells": [
                    {
                        "angular_momentum": [0, 1],
                        "exponents": [1],
                        "coefficients": [[1]],
                    }
                ]
            },
            "one coefficient row",
        ),
        (
            {
                "electron_shells": [
                    {
                        "angular_momentum": [0],
                        "exponents": [1, 0.5],
                        "coefficients": [[1]],
                    }
                ]
            },
            "Invalid basis shell",
        ),
        (
            {
                "electron_shells": [
                    {"angular_momentum": [0], "exponents": [-1], "coefficients": [[1]]}
                ]
            },
            "Invalid basis shell",
        ),
    ],
)
def test_invalid_basis_and_unsupported_physics_are_rejected(change, message):
    data = copy.deepcopy(HYDROGEN)
    data["elements"]["1"].update(change)
    with pytest.raises(ValueError, match=message):
        gi.Molecule.from_xyz(XYZ, data)


def test_input_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        gi.Molecule.from_xyz(tmp_path / "missing.xyz", HYDROGEN)
    with pytest.raises(FileNotFoundError):
        gi.Molecule.from_xyz(XYZ, tmp_path / "missing.gbs")
    with pytest.raises(ValueError, match="unit"):
        gi.Molecule.from_xyz(XYZ, HYDROGEN, unit="nm")
    with pytest.raises(ValueError, match="basis_format"):
        gi.Molecule.from_xyz(XYZ, HYDROGEN, basis_format="invalid")
    with pytest.raises(ValueError, match="no data for element He"):
        gi.Molecule([("He", (0, 0, 0))], HYDROGEN)
    with pytest.raises(ValueError, match="elements"):
        gi.Molecule.from_xyz(XYZ, {})
    with pytest.raises(ValueError, match="Invalid BSE JSON"):
        gi.Molecule.from_xyz(XYZ, "{bad}")
    with pytest.raises(TypeError, match="MagneticField"):
        gi.Molecule.from_xyz(XYZ, HYDROGEN, field=(0, 0, 1))
    with pytest.raises(ValueError, match="at least one atom"):
        gi.Molecule([], HYDROGEN)


@pytest.mark.parametrize(
    "fmt, suffix", [("gaussian94", ".gbs"), ("nwchem", ".nw"), ("json", ".json")]
)
def test_named_basis_and_standard_file_roundtrips(fmt, suffix, tmp_path):
    bse = pytest.importorskip("basis_set_exchange")
    xyz = "3\nwater\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24\n"
    named = gi.Molecule.from_xyz(xyz, "cc-pVDZ")
    text = bse.get_basis("cc-pVDZ", elements=[1, 8], fmt=fmt)
    path = tmp_path / ("basis" + suffix)
    path.write_text(text)
    assert named.ao_count == 25  # Cartesian d shell even for spherical source data.
    for source, format_arg in [(path, None), (str(path), None), (text, fmt)]:
        loaded = gi.Molecule.from_xyz(xyz, source, basis_format=format_arg)
        assert loaded.ao_count == named.ao_count
        np.testing.assert_allclose(loaded.overlap(), named.overlap(), atol=2e-13)
        np.testing.assert_allclose(loaded.kinetic(), named.kinetic(), atol=2e-12)


def test_named_basis_errors_and_ecp():
    pytest.importorskip("basis_set_exchange")
    with pytest.raises(ValueError, match="Could not load basis"):
        gi.Molecule.from_xyz(XYZ, "not-a-basis")
    with pytest.raises(ValueError, match="ECP"):
        gi.Molecule([("Cu", (0, 0, 0))], "LANL2DZ")


@pytest.mark.parametrize(
    "fmt, text",
    [
        ("gaussian94", "****\nH 0\nS 1 1.00\nnot-a-number 1.0\n****\n"),
        ("nwchem", 'BASIS "ao basis" PRINT\nH S\nnot-a-number 1.0\nEND\n'),
    ],
)
def test_malformed_formatted_basis_is_diagnostic(fmt, text):
    pytest.importorskip("basis_set_exchange")
    with pytest.raises(ValueError, match=f"Invalid {fmt} basis"):
        gi.Molecule.from_xyz(XYZ, text, basis_format=fmt)


def test_explicit_format_for_nonstandard_filename(tmp_path):
    bse = pytest.importorskip("basis_set_exchange")
    path = tmp_path / "custom.txt"
    path.write_text(bse.get_basis("STO-3G", elements=[1], fmt="gaussian94"))
    with pytest.raises(ValueError, match="Cannot infer"):
        gi.Molecule.from_xyz(XYZ, path)
    assert gi.Molecule.from_xyz(XYZ, path, basis_format="gaussian94").ao_count == 2
