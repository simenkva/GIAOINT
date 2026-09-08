# References and use notes

Only bibliographic facts and equations were consulted. No external source code
has been copied into this repository.

1. F. London, “Théorie quantique des courants interatomiques dans les
   combinaisons aromatiques,” *Journal de Physique et le Radium* **8**,
   397–409 (1937). DOI:
   [10.1051/jphysrad:01937008010039700](https://doi.org/10.1051/jphysrad:01937008010039700).
   Historical source for London orbitals.
2. T. Helgaker and P. Jørgensen, “An electronic Hamiltonian for origin
   independent calculations of magnetic properties,” *J. Chem. Phys.* **95**,
   2595–2601 (1991). DOI:
   [10.1063/1.460912](https://doi.org/10.1063/1.460912). Used to check
   gauge-origin transformation conventions.
3. E. I. Tellgren, A. Soncini, and T. Helgaker, “Nonperturbative ab initio
   calculations in strong magnetic fields using London orbitals,”
   *J. Chem. Phys.* **129**, 154114 (2008). DOI:
   [10.1063/1.2996525](https://doi.org/10.1063/1.2996525). Primary source for
   the frozen \(e^{-i\kappa\cdot r}\) convention, London overlap distributions,
   the MD and OS generalizations, complex product centers, and complex Boys
   arguments.
4. E. I. Tellgren, S. S. Reine, and T. Helgaker, “Analytical GIAO and
   hybrid-basis integral derivatives: application to geometry optimization of
   molecules in strong magnetic fields,” *Phys. Chem. Chem. Phys.* **14**,
   9492–9499 (2012). DOI:
   [10.1039/C2CP40965H](https://doi.org/10.1039/C2CP40965H). Used to confirm
   that derivatives can be represented through modified Gaussian/plane-wave
   functions and must include phase derivatives.
5. L. E. McMurchie and E. R. Davidson, “One- and two-electron integrals over
   Cartesian Gaussian functions,” *J. Comput. Phys.* **26**, 218–231 (1978).
   DOI: [10.1016/0021-9991(78)90092-X](https://doi.org/10.1016/0021-9991(78)90092-X).
   Source for the field-free Hermite expansion and auxiliary-function design.
6. S. Obara and A. Saika, “Efficient recursive computation of molecular
   integrals over Cartesian Gaussian functions,” *J. Chem. Phys.* **84**,
   3963–3974 (1986). DOI:
   [10.1063/1.450106](https://doi.org/10.1063/1.450106). Source for the direct
   Cartesian recurrence alternative.
7. M. Head-Gordon and J. A. Pople, “A method for two-electron Gaussian integral
   and integral derivative evaluation using recurrence relations,”
   *J. Chem. Phys.* **89**, 5777–5786 (1988). DOI:
   [10.1063/1.455553](https://doi.org/10.1063/1.455553). Source for moving
   horizontal recurrence work outside contraction loops and the later
   optimized-backend candidate.
8. M. Dupuis, J. Rys, and H. F. King, “Evaluation of molecular integrals over
   Gaussian basis functions,” *J. Chem. Phys.* **65**, 111–116 (1976). DOI:
   [10.1063/1.432807](https://doi.org/10.1063/1.432807). Source for the Rys
   quadrature alternative.
9. M. Tachikawa and M. Shiga, “Evaluation of atomic integrals for hybrid
   Gaussian type and plane-wave basis functions via the McMurchie-Davidson
   recursion formula,” *Phys. Rev. E* **64**, 056706 (2001). DOI:
   [10.1103/PhysRevE.64.056706](https://doi.org/10.1103/PhysRevE.64.056706).
   Independent precedent for MD with Gaussian/plane-wave hybrid functions.
10. G. Beylkin and S. Sharma, “A fast algorithm for computing the Boys
    function,” *J. Chem. Phys.* **155**, 174117 (2021). DOI:
    [10.1063/5.0062444](https://doi.org/10.1063/5.0062444). Primary numerical
    source for real and complex Boys evaluation, recurrence direction, and
    negative-real-part handling.
