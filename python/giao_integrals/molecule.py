"""Molecules from XYZ and standard Gaussian basis sets, with NumPy integrals."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _giao_integrals as core
from .eri import EriBatch
from .eri import eri as compute_eri
from .eri import eri_batches as compute_eri_batches

# CODATA 2022, https://physics.nist.gov/cuu/pdf/wall_2022.pdf
BOHR_IN_ANGSTROM = 0.529177210544
_DEFAULT_ERI_BYTES = 512 * 1024**2
_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au "
    "Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf "
    "Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_ATOMIC_NUMBERS = {symbol: z for z, symbol in enumerate(_SYMBOLS, start=1)}
_BASIS_SUFFIXES = {
    ".json": "json",
    ".gbs": "gaussian94",
    ".g94": "gaussian94",
    ".nw": "nwchem",
    ".nwchem": "nwchem",
}

Atom = tuple[str, tuple[float, float, float]]
BasisSource = str | PathLike[str] | Mapping[str, Any]
Matrix = npt.NDArray[np.complex128]


def _number(value: Any) -> float:
    return float(
        value.replace("D", "E").replace("d", "e") if isinstance(value, str) else value
    )


def _symbol(value: str) -> str:
    symbol = str(value).strip().capitalize()
    if symbol.isdecimal() and 1 <= int(symbol) <= len(_SYMBOLS):
        symbol = _SYMBOLS[int(symbol) - 1]
    if symbol not in _ATOMIC_NUMBERS:
        raise ValueError(
            f"Unknown element {value!r}; use an element symbol or atomic number"
        )
    return symbol


def _atoms_in_bohr(
    atoms: Iterable[tuple[str, Sequence[float]]], unit: str
) -> tuple[Atom, ...]:
    factors = {"angstrom": 1.0 / BOHR_IN_ANGSTROM, "bohr": 1.0}
    if unit not in factors:
        raise ValueError("unit must be 'angstrom' or 'bohr'")
    result = []
    for index, atom in enumerate(atoms, start=1):
        try:
            symbol, coordinates = atom
            position = tuple(_number(x) * factors[unit] for x in coordinates)
            if len(position) != 3 or not all(math.isfinite(x) for x in position):
                raise ValueError("coordinates must be three finite numbers")
            result.append((_symbol(symbol), position))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid atom {index}: {error}") from error
    if not result:
        raise ValueError("A molecule must contain at least one atom")
    return tuple(result)


def _read_xyz(source: str | PathLike[str]) -> list[tuple[str, list[str]]]:
    if isinstance(source, PathLike) or "\n" not in source and "\r" not in source:
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        text = source.lstrip("\ufeff")
    lines = text.lstrip("\r\n").splitlines()
    try:
        count = int(lines[0].strip())
    except (IndexError, ValueError) as error:
        raise ValueError("XYZ must start with an integer atom count") from error
    if count < 1:
        raise ValueError("XYZ atom count must be positive")
    if len(lines) < count + 2:
        raise ValueError(f"XYZ requires a comment line and {count} atom lines")
    if any(line.strip() for line in lines[count + 2 :]):
        raise ValueError(
            "Expected one XYZ frame; extra atoms or frames follow the declared count"
        )
    atoms = []
    for index, line in enumerate(lines[2 : count + 2], start=3):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(
                f"XYZ line {index} must contain an element and three coordinates"
            )
        atoms.append((parts[0], parts[1:]))
    return atoms


def _basis_set_exchange():
    try:
        import basis_set_exchange
    except ImportError as error:
        raise ImportError(
            "Named, Gaussian94, and NWChem bases require basis-set-exchange. "
            "Install it with: pip install 'giao-integrals[molecule]' "
            "(or pip install '.[molecule]' from this repository). "
            "BSE JSON input works without this extra."
        ) from error
    return basis_set_exchange


def _basis_data(
    source: BasisSource, symbols: Sequence[str], fmt: str | None
) -> Mapping[str, Any]:
    if fmt is not None and fmt not in {"json", "gaussian94", "nwchem"}:
        raise ValueError("basis_format must be 'json', 'gaussian94', or 'nwchem'")
    if isinstance(source, Mapping):
        if fmt not in (None, "json"):
            raise ValueError("A basis dictionary uses basis_format='json'")
        return source
    if not isinstance(source, (str, PathLike)):
        raise TypeError(
            "basis must be a basis name, file path, formatted text, or BSE dictionary"
        )
    inline = isinstance(source, str) and (
        "\n" in source or "\r" in source or source.lstrip().startswith("{")
    )
    if inline:
        text = source
        fmt = fmt or ("json" if text.lstrip().startswith("{") else None)
        if fmt is None:
            raise ValueError(
                "Specify basis_format for inline Gaussian94 or NWChem text"
            )
    else:
        path = Path(source)
        is_path = (
            isinstance(source, PathLike)
            or path.is_file()
            or path.suffix.lower() in _BASIS_SUFFIXES
            or "/" in str(source)
            or "\\" in str(source)
            or fmt is not None
        )
        if not is_path:
            bse = _basis_set_exchange()
            try:
                return bse.get_basis(str(source), elements=sorted(set(symbols)))
            except (KeyError, ValueError) as error:
                raise ValueError(f"Could not load basis {source!r}: {error}") from error
        text = path.read_text(encoding="utf-8-sig")
        fmt = fmt or _BASIS_SUFFIXES.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(
                "Cannot infer basis format; specify basis_format="
                "'gaussian94', 'nwchem', or 'json'"
            )
    if fmt == "json":
        try:
            data = json.loads(text)
        except ValueError as error:
            raise ValueError(f"Invalid BSE JSON basis: {error}") from error
        if not isinstance(data, Mapping):
            raise ValueError("BSE JSON basis must be an object containing 'elements'")
        return data
    bse = _basis_set_exchange()
    from jsonschema.exceptions import ValidationError

    try:
        return bse.read_formatted_basis_str(text, fmt, validate=True)
    except (KeyError, ValueError, RuntimeError, IndexError, ValidationError) as error:
        raise ValueError(f"Invalid {fmt} basis: {error}") from error


def _build_basis(
    atoms: tuple[Atom, ...], data: Mapping[str, Any]
) -> tuple[core.Basis, tuple[int, ...]]:
    elements = data.get("elements")
    if not isinstance(elements, Mapping):
        raise ValueError(
            "BSE basis data must contain an 'elements' object keyed by atomic number"
        )
    shells = []
    owners = []
    for atom_index, (symbol, position) in enumerate(atoms):
        element = elements.get(str(_ATOMIC_NUMBERS[symbol]))
        if not isinstance(element, Mapping):
            raise ValueError(f"Basis has no data for element {symbol}")
        if element.get("ecp_potentials") or element.get("ecp_electrons", 0):
            raise ValueError(
                f"Basis for {symbol} contains an ECP; "
                "only all-electron bases are supported"
            )
        entries = element.get("electron_shells")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Basis for {symbol} has no electron shells")
        specifications = []
        for index, entry in enumerate(entries):
            try:
                if entry.get("function_type", "gto") not in {
                    "gto",
                    "gto_cartesian",
                    "gto_spherical",
                }:
                    raise ValueError("only Gaussian orbital shells are supported")
                angular = entry["angular_momentum"]
                if not angular or any(type(x) is not int or x < 0 for x in angular):
                    raise ValueError("angular momenta must be non-negative integers")
                if len(set(angular)) != len(angular):
                    raise ValueError(
                        "combined shells must have distinct angular momenta"
                    )
                exponents = [_number(x) for x in entry["exponents"]]
                coefficients = [
                    [_number(x) for x in row] for row in entry["coefficients"]
                ]
                if len(angular) == 1:
                    specifications.append((angular[0], exponents, coefficients))
                elif len(coefficients) == len(angular):
                    specifications.extend(
                        (momentum, exponents, [row])
                        for momentum, row in zip(angular, coefficients, strict=True)
                    )
                else:
                    raise ValueError(
                        "combined shells require one coefficient row "
                        "per angular momentum"
                    )
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
            ) as error:
                raise ValueError(
                    f"Invalid basis shell {index} for {symbol}: {error}"
                ) from error
        # Preserve atom order, then stable increasing L order within each atom.
        # A general contraction remains one shell with multiple coefficient rows.
        for angular, exponents, coefficients in sorted(
            specifications, key=lambda x: x[0]
        ):
            try:
                shells.append(core.Shell(position, angular, exponents, coefficients))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"Invalid basis shell for {symbol}: {error}"
                ) from error
            owners.append(atom_index)
    return core.Basis(shells), tuple(owners)


@dataclass(frozen=True)
class MolecularIntegrals:
    """Cartesian complex128 matrices in atomic units; ERIs are optional.

    ``kinetic`` is canonical kinetic energy. ``core_hamiltonian`` includes
    physical magnetic kinetic energy plus nuclear attraction, without spin.
    """

    overlap: Matrix
    kinetic: Matrix
    nuclear_attraction: Matrix
    core_hamiltonian: Matrix
    eri: Matrix | None = None


class Molecule:
    """Geometry, a Cartesian basis, and integral methods.

    Construct from ``[(symbol, (x, y, z)), ...]`` or use ``from_xyz`` for
    XYZ files/text. Coordinates default to angstrom; stored atoms and nuclei
    are in bohr. ``basis`` accepts a BSE name, file, text, or dictionary.
    Named bases and Gaussian94/NWChem parsing use the optional ``molecule``
    extra. BSE JSON needs no optional dependency. No network requests occur.

    Output is always Cartesian, including when a source basis declares
    spherical shells. ECPs and non-Gaussian shells are unsupported.
    """

    def __init__(
        self,
        atoms: Iterable[tuple[str, Sequence[float]]],
        basis: BasisSource,
        *,
        unit: str = "angstrom",
        basis_format: str | None = None,
        field: core.MagneticField | None = None,
    ) -> None:
        if field is not None and not isinstance(field, core.MagneticField):
            raise TypeError("field must be a MagneticField or None")
        self._atoms = _atoms_in_bohr(atoms, unit)
        data = _basis_data(basis, [symbol for symbol, _ in self._atoms], basis_format)
        self._basis, self._shell_atoms = _build_basis(self._atoms, data)
        self._nuclei = tuple(
            core.Nucleus(_ATOMIC_NUMBERS[symbol], position)
            for symbol, position in self._atoms
        )
        self._field = field if field is not None else core.MagneticField()

    @classmethod
    def from_xyz(
        cls,
        source: str | PathLike[str],
        basis: BasisSource,
        *,
        unit: str = "angstrom",
        basis_format: str | None = None,
        field: core.MagneticField | None = None,
    ) -> Molecule:
        """Read one standard XYZ frame from a path or multiline text.

        The second line is a required comment (possibly blank). Atom rows
        contain an element symbol/atomic number and three coordinates.
        Use ``unit='bohr'`` for files whose coordinates are already in bohr.
        """
        return cls(
            _read_xyz(source), basis, unit=unit, basis_format=basis_format, field=field
        )

    @property
    def atoms(self) -> tuple[Atom, ...]:
        """Element symbols and coordinates in bohr, in input order."""
        return self._atoms

    @property
    def basis(self) -> core.Basis:
        """The low-level basis, usable with all existing integral APIs."""
        return self._basis

    @property
    def nuclei(self) -> tuple[core.Nucleus, ...]:
        return self._nuclei

    @property
    def shell_atoms(self) -> tuple[int, ...]:
        """Zero-based owner atom for each basis shell."""
        return self._shell_atoms

    @property
    def field(self) -> core.MagneticField:
        """Field in atomic units, with gauge origin in bohr."""
        return self._field

    @property
    def ao_count(self) -> int:
        return self._basis.ao_count

    def __repr__(self) -> str:
        return (
            f"Molecule(atoms={len(self.atoms)}, shells={len(self.shell_atoms)}, "
            f"cartesian_aos={self.ao_count})"
        )

    def overlap(self) -> Matrix:
        return core.overlap(self.basis, field=self.field)

    def kinetic(self) -> Matrix:
        """Canonical kinetic energy, -1/2 Laplacian."""
        return core.kinetic(self.basis, field=self.field)

    def magnetic_kinetic(self) -> Matrix:
        """Physical orbital kinetic energy in the magnetic field (no spin)."""
        return core.magnetic_kinetic(self.basis, field=self.field)

    def nuclear_attraction(self) -> Matrix:
        return core.nuclear_attraction(self.basis, self.nuclei, field=self.field)

    def core_hamiltonian(self) -> Matrix:
        """Physical magnetic kinetic energy plus nuclear attraction."""
        return self.magnetic_kinetic() + self.nuclear_attraction()

    def eri(
        self,
        *,
        max_bytes: int = _DEFAULT_ERI_BYTES,
        screening_threshold: float = 0.0,
        threads: int = 1,
    ) -> Matrix:
        """Return a full (nao, nao, nao, nao) tensor, capped at 512 MiB by default.

        Use ``eri_batches()`` for streaming larger calculations. Screening
        and parallelism are opt-in, as in the low-level API.
        """
        return compute_eri(
            self.basis,
            field=self.field,
            storage="full",
            max_bytes=max_bytes,
            screening_threshold=screening_threshold,
            threads=threads,
        )

    def eri_batches(
        self,
        *,
        target_bytes: int = 64 * 1024**2,
        screening_threshold: float = 0.0,
        threads: int = 1,
    ) -> Iterable[EriBatch]:
        """Stream canonical shell-quartet batches with the molecule's field."""
        return compute_eri_batches(
            self.basis,
            field=self.field,
            target_bytes=target_bytes,
            screening_threshold=screening_threshold,
            threads=threads,
        )

    def integrals(
        self,
        *,
        eri: bool = False,
        max_eri_bytes: int = _DEFAULT_ERI_BYTES,
        screening_threshold: float = 0.0,
        threads: int = 1,
    ) -> MolecularIntegrals:
        """Compute S, canonical T, V, and physical Hcore; optionally full ERIs."""
        overlap = self.overlap()
        kinetic = self.kinetic()
        attraction = self.nuclear_attraction()
        physical = (
            kinetic if all(x == 0.0 for x in self.field.B) else self.magnetic_kinetic()
        )
        return MolecularIntegrals(
            overlap,
            kinetic,
            attraction,
            physical + attraction,
            (
                self.eri(
                    max_bytes=max_eri_bytes,
                    screening_threshold=screening_threshold,
                    threads=threads,
                )
                if eri
                else None
            ),
        )
