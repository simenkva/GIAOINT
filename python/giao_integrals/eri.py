"""Memory-bounded Python interfaces for electron-repulsion integrals."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np
import numpy.typing as npt

from ._giao_integrals import (
    Basis,
    MagneticField,
    _eri_batch,
    _eri_full,
    _EriSchwarzBounds,
)

Quartet = tuple[int, int, int, int]


@dataclass(frozen=True)
class EriBatch:
    """Packed shell-quartet results and screening counts.

    ``values[offsets[i]:offsets[i + 1]]`` reshaped to ``shapes[i]`` is the
    C-contiguous block for ``quartets[i]``.
    """

    quartets: npt.NDArray[np.uint32]
    shapes: npt.NDArray[np.int64]
    offsets: npt.NDArray[np.int64]
    values: npt.NDArray[np.complex128]
    requested_count: int = 0
    screened_count: int = 0

    def block(self, index: int) -> npt.NDArray[np.complex128]:
        """Return one shell-quartet block as a shaped view."""

        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.values[start:stop].reshape(tuple(self.shapes[index]))


def _checked_quartets(
    shell_count: int, quartets: Iterable[Sequence[int]] | None
) -> Iterator[Quartet]:
    if quartets is None:
        for quartet in product(range(shell_count), repeat=4):
            a, b, c, d = quartet
            orbit = (quartet, (c, d, a, b), (b, a, d, c), (d, c, b, a))
            if min(orbit) == quartet:
                yield quartet
        return

    for value in quartets:
        if len(value) != 4:
            raise ValueError("each shell quartet must contain four indices")
        if any(isinstance(index, (bool, np.bool_)) for index in value):
            raise ValueError("shell-quartet indices must be integers")
        try:
            quartet = tuple(int(index) for index in value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("shell-quartet indices must be integers") from error
        if any(
            converted != original
            for converted, original in zip(quartet, value, strict=True)
        ):
            raise ValueError("shell-quartet indices must be integers")
        if any(index < 0 or index >= shell_count for index in quartet):
            raise IndexError("shell-quartet index is out of range")
        yield quartet  # type: ignore[misc]


def eri_batches(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    quartets: Iterable[Sequence[int]] | None = None,
    target_bytes: int = 64 * 1024 * 1024,
    screening_threshold: float = 0.0,
    threads: int = 1,
) -> Iterator[EriBatch]:
    """Yield packed shell-quartet batches without allocating an AO tensor.

    With no explicit ``quartets``, only canonical representatives under the
    exact finite-field ERI symmetries are emitted. Screening is disabled when
    ``screening_threshold`` is zero, which is the default.
    """

    if isinstance(target_bytes, (bool, np.bool_)) or not isinstance(
        target_bytes, (int, np.integer)
    ):
        raise ValueError("target_bytes must be a positive integer")
    if target_bytes <= 0:
        raise ValueError("target_bytes must be a positive integer")
    screening_threshold = float(screening_threshold)
    if not np.isfinite(screening_threshold) or screening_threshold < 0.0:
        raise ValueError("screening_threshold must be finite and non-negative")
    if isinstance(threads, (bool, np.bool_)) or not isinstance(
        threads, (int, np.integer)
    ):
        raise ValueError("threads must be a positive integer")
    if threads <= 0:
        raise ValueError("threads must be a positive integer")

    shells = basis.shells
    bounds = (
        _EriSchwarzBounds(basis, field=field) if screening_threshold > 0.0 else None
    )
    packed_quartets: list[Quartet] = []
    packed_bytes = 0

    def emit() -> EriBatch:
        quartet_array, shapes, offsets, values, requested, screened = _eri_batch(
            basis,
            packed_quartets,
            field=field,
            screening_threshold=screening_threshold,
            threads=int(threads),
            bounds=bounds,
        )
        return EriBatch(
            np.asarray(quartet_array, dtype=np.uint32),
            np.asarray(shapes, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
            np.asarray(values, dtype=np.complex128),
            int(requested),
            int(screened),
        )

    for quartet in _checked_quartets(len(shells), quartets):
        block_bytes = (
            prod(shells[index].ao_count for index in quartet)
            * np.dtype(np.complex128).itemsize
        )
        if packed_quartets and packed_bytes + block_bytes > target_bytes:
            yield emit()
            packed_quartets.clear()
            packed_bytes = 0
        packed_quartets.append(quartet)
        packed_bytes += int(block_bytes)

    if packed_quartets:
        yield emit()


def eri(
    basis: Basis,
    *,
    field: MagneticField | None = None,
    storage: str = "blocks",
    max_bytes: int | None = None,
    target_bytes: int = 64 * 1024 * 1024,
    screening_threshold: float = 0.0,
    threads: int = 1,
) -> Iterator[EriBatch] | npt.NDArray[np.complex128]:
    """Compute ERIs as streamed blocks or an explicitly guarded full tensor."""

    screening_threshold = float(screening_threshold)
    if not np.isfinite(screening_threshold) or screening_threshold < 0.0:
        raise ValueError("screening_threshold must be finite and non-negative")
    if isinstance(threads, (bool, np.bool_)) or not isinstance(
        threads, (int, np.integer)
    ):
        raise ValueError("threads must be a positive integer")
    if threads <= 0:
        raise ValueError("threads must be a positive integer")
    if storage == "blocks":
        if max_bytes is not None:
            raise ValueError("max_bytes is only valid with storage='full'")
        return eri_batches(
            basis,
            field=field,
            target_bytes=target_bytes,
            screening_threshold=screening_threshold,
            threads=int(threads),
        )
    if storage != "full":
        raise ValueError("storage must be 'blocks' or 'full'")
    if max_bytes is None:
        raise ValueError("storage='full' requires an explicit max_bytes limit")
    if isinstance(max_bytes, (bool, np.bool_)) or not isinstance(
        max_bytes, (int, np.integer)
    ):
        raise ValueError("max_bytes must be a non-negative integer")
    if max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    count = basis.ao_count
    required = count**4 * np.dtype(np.complex128).itemsize
    if required > max_bytes:
        raise MemoryError(
            f"full ERI tensor requires {required} bytes, "
            f"exceeding max_bytes={max_bytes}"
        )
    return np.asarray(
        _eri_full(
            basis,
            field=field,
            screening_threshold=screening_threshold,
            threads=int(threads),
        ),
        dtype=np.complex128,
    )


def eri_schwarz_bounds(
    basis: Basis, *, field: MagneticField | None = None
) -> npt.NDArray[np.float64]:
    """Return the shell-pair Schwarz factors used by ERI screening."""

    return np.asarray(_EriSchwarzBounds(basis, field=field).values, dtype=np.float64)
