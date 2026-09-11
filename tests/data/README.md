# Frozen ERI regression data

`eri_recursive_m8.jsonl` contains 96 input quartets and 192 expected complex
values from the recursive M8 extension, captured before the M9 rewrite at
commit `2d5b3b13c85fb1258788d698bff551bebbde4b93`. Each line includes both a
zero-field result and a finite-field result. Inputs came from Python's
`random.Random(9001)`: angular totals 0 through 4 on each center, exponents
0.4 through 2, centers in [-0.8, 0.8], signed coefficients in [-1, 1], and
fields in [-1.2, 1.2]. Normalization alternates between quartets.

Keep these values tied to the recursive baseline. Do not regenerate them
with the optimized kernel to accommodate a failure. The independent
high-precision OS tests provide the separate mathematical oracle.
