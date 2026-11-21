"""Time bounded, parallel derivative batches on a 105-AO basis.

    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        python benchmarks/eri_derivative_stream.py --threads 1 4

Uses a specified representative subset, not the full 105**4 derivative tensor:
40 ordered quartets spanning every shell, repeated 20 times by default.
Basis construction and hashing/equality checks are outside the measured time;
generator packing, optional screening-bound setup and native evaluation are in it.
"""

import argparse
import hashlib
from time import perf_counter

import giao_integrals as gi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", nargs="+", type=int, default=[1])
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--target-bytes", type=int, default=1024**2)
    parser.add_argument("--screening-threshold", type=float, default=0.0)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    mol = gi.Molecule(
        [("Ne", (0.1, -0.2, 0.3))],
        "aug-cc-pVQZ",
        unit="bohr",
        field=gi.MagneticField((0.05, -0.03, 0.2), (0.37, -0.29, 0.43)),
    )
    quartets = [
        q
        for i in range(len(mol.basis.shells))
        for q in [(i, 1, 3, 1), (1, i, 1, 3), (3, 1, i, 1), (1, 3, 1, i)]
    ] * args.repeat
    dense = mol.ao_count**4 * 12 * 16 / 1024**3
    print(
        f"Ne/aug-cc-pVQZ: {mol.ao_count} AOs; "
        f"{len(quartets)} explicit quartets; "
        f"full derivative estimate {dense:.3f} GiB",
        flush=True,
    )
    reference = None
    for threads in args.threads:
        iterator = iter(
            mol.eri_derivative_batches(
                quartets=quartets,
                target_bytes=args.target_bytes,
                screening_threshold=args.screening_threshold,
                threads=threads,
            )
        )
        seconds = 0.0
        peak = 0
        screened = 0
        digest = hashlib.sha256()
        while True:
            start = perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                seconds += perf_counter() - start
                break
            seconds += perf_counter() - start
            peak = max(peak, batch.values.nbytes)
            screened += batch.screened_count
            digest.update(memoryview(batch.quartets))
            digest.update(memoryview(batch.values))
        result = digest.hexdigest()
        if reference is None:
            reference = result
        elif result != reference:
            raise RuntimeError("threaded derivative batches differ")
        print(
            f"threads={threads}, seconds={seconds:.6f}, "
            f"peak packed MiB={peak/1024**2:.3f}, "
            f"screened={screened}, identical=yes",
            flush=True,
        )


if __name__ == "__main__":
    main()
