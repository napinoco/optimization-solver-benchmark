# DIMACS / SDPLIB Known Objective Values

This note documents the source and caveats of the `known_objective_value`
entries for DIMACS and SDPLIB problems in `config/problem_registry.yaml`.
See [NETLIB_KNOWN_OBJECTIVES.md](NETLIB_KNOWN_OBJECTIVES.md) for the
equivalent note covering NETLIB (which also documents a currently unfixed
data quality bug — worth reading if you're comparing solver results against
known values).

## DIMACS

### Source

The values come from the DIMACS library's own problem table, published by
the 7th DIMACS Implementation Challenge (compiled by Gabor Pataki and
Stefan H. Schmieta), originally hosted at
[dimacs.rutgers.edu](http://dimacs.rutgers.edu/archive/Challenges/Seventh/Instances/)
and mirrored in this project's `problems/DIMACS` submodule as `README.md`.

### Caveats from the source

- **torus set sign/scale convention**: the `max{c'x}` problems in this set
  are solved here as `min{-c'x}`, so the table's "Opt. value" must be
  multiplied by `-1` to match what a SeDuMi-form solve returns; `torusg*`
  values must additionally be divided by 100,000. This is why
  `torusg3-8`, `torusg3-15`, `toruspm3-8-50`, and `toruspm3-15-50` carry a
  negative `known_objective_value` in the registry (corrected in commit
  `b8b5ec2`). The `hamming_*` theta-function values were corrected to
  negative signs in the same commit, though the source README does not
  spell out an explicit min/max caveat for that set the way it does for
  torus — treat the sign as empirically matched rather than derived from a
  documented rule.
- **`fap` set lower bounds**: `fap25`, `fap-sup25`, and `fap-sup36` are
  marked `(lb, not opt)` in the source table — the listed value is a lower
  bound, not a proven optimum. `fap25` is intentionally kept in the
  registry without a `known_objective_value` for this reason; `fap36` and
  the `fap-sup*` problems are currently commented out of the registry
  entirely ("Solver issues", commit `b8b5ec2`).
- **`hinf12` / `hinf13` uncertain accuracy**: marked `(?)` in the source
  table — "the listed value is the currently known most accurate one;
  nevertheless, its accuracy is still not satisfactory, and the true value
  may be quite different." Both are present in the registry with numeric
  `known_objective_value` fields; treat accuracy comparisons against them
  with correspondingly reduced confidence.
- **`biomedP`, `industry2`**: excluded from the registry (source files are
  `.dat`, not the supported `.dat-s`/`.mat` formats).

## SDPLIB

### Source

The values come from the SDPLIB library's own problem table, Brian
Borchers' SDPLIB 1.2 (Borchers, B., *SDPLIB 1.2, A Library of Semidefinite
Programming Test Problems*, Optimization Methods and Software, 11(1):683-690,
1999), mirrored in this project's `problems/SDPLIB` submodule as `README.md`.

### Caveats from the source

- **Primal/dual sign convention**: "different authors have adopted
  different conventions for the primal and dual SDP problems... thus some
  objective function values have their signs changed" relative to other
  conventions; the source table's values follow SDPA's convention. Loading
  SDPLIB here goes through a SeDuMi-form conversion, and a batch of sign
  corrections across 88 problems was needed to match (commit `cea8c62`,
  "Fix SDPLIB objective value signs due to SDPA to SeDuMi format
  conversion").
- **`infd1`, `infd2`, `infp1`, `infp2`**: reported as "dual infeasible" /
  "primal infeasible" rather than a numeric objective value; these are not
  currently in the registry (no numeric `known_objective_value` applies).
- **Conversion precision**: "in some cases, very slight changes in the
  optimal objective function value have occurred as a result of the
  conversion into SDPA format" — treat the published values as reference
  points for sanity-checking solver output, not exact ground truth to many
  digits.

## Related files

- [`config/problem_registry.yaml`](../../config/problem_registry.yaml) — the `known_objective_value` fields themselves
- `problems/DIMACS/README.md` and `problems/SDPLIB/README.md` — the source tables (populated once the submodules are checked out, e.g. `git submodule update --init problems/DIMACS problems/SDPLIB`)
