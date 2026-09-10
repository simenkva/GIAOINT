"""Consume memory-bounded canonical ERI shell-quartet batches."""

from __future__ import annotations

import giao_integrals as gi
import numpy as np


def main() -> None:
    basis = gi.Basis(
        [
            gi.Shell((-0.7, 0.0, 0.0), 0, [1.0, 0.3], [0.6, 0.5]),
            gi.Shell((0.7, 0.0, 0.0), 0, [1.0, 0.3], [0.6, 0.5]),
        ]
    )
    field = gi.MagneticField((0.03, -0.01, 0.02))

    block_count = 0
    checksum = 0.0j
    for batch in gi.eri_batches(basis, field=field, target_bytes=128):
        for index in range(len(batch.quartets)):
            block = batch.block(index)
            assert block.dtype == np.complex128 and block.flags.c_contiguous
            checksum += block.sum()
            block_count += 1

    print(f"consumed {block_count} canonical blocks; checksum={checksum}")


if __name__ == "__main__":
    main()
