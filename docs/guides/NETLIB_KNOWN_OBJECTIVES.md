# NETLIB Known Objective Values

This note documents the source and caveats of the `known_objective_value`
entries for NETLIB problems in `config/problem_registry.yaml`, and a data
quality issue discovered while validating them.

## Source

The values come from the official NETLIB LP problem summary table published
at [`netlib.org/lp/data/readme`](https://www.netlib.org/lp/data/readme),
compiled by Bob Bixby. Per that document:

> "The optimal values are from MINOS version 5.3 (of Sept. 1988) running on
> a VAX with default options."

### Caveats from the source

- **Solver/platform sensitivity**: MINOS control parameters (SCALE, PARTIAL
  PRICE, FEASIBILITY TOLERANCE, OPTIMALITY TOLERANCE, CRASH OPTION), the
  MINOS version, the computer, and even the compiler used can all affect the
  reported value.
- **Known solver discrepancies**: Bob Bixby reports that CPLEX (on a Sparc
  station) finds slightly different optimal values for some problems.
- **`standgub` has no reported value**: `STANDGUB` includes GUB (generalized
  upper bound) markers that MINOS does not understand. Per the source
  readme, removing those marker rows (`EGROUP`/`ENDX`) makes `STANDGUB`
  identical to `STANDATA`. `config/problem_registry.yaml` intentionally
  leaves `standgub` without a `known_objective_value`.

### Name mapping

A few registry keys differ from the NETLIB table's names (dots/hyphens
aren't valid YAML keys, and a couple of names collide with existing
DIMACS/SDPLIB problems):

| Registry key | NETLIB name |
|---|---|
| `netlib_25fv47` | `25FV47` |
| `netlib_80bau3b` | `80BAU3B` |
| `netlib_truss` | `TRUSS` |
| `gfrd_pnc` | `GFRD-PNC` |
| `maros_r7` | `MAROS-R7` |
| `pilot_ja` | `PILOT.JA` |
| `pilot_we` | `PILOT.WE` |
| `vtp_base` | `VTP.BASE` |

All other registry keys match the NETLIB name case-insensitively.

## Data quality finding: free-variable problems

While cross-checking benchmark results against these known values, a
pre-existing bug was found in `scripts/data_loaders/python/mps_loader.py`
(and, by inheritance, in `scripts/solvers/python/scipy_runner.py`'s and
CVXPY's bounds/cone reconstruction): variables are **not reordered** so that
free variables (MPS `FR`/`MI` bounds) come first, even though the
`cone_structure` the loader emits (`free_vars` / `nonneg_vars` counts) is
consumed downstream *assuming* that ordering. As a result, whichever
variables happen to occupy the first `free_vars` positions in the original
MPS column order are treated as free, and the rest as nonnegative —
frequently the wrong assignment.

This was confirmed by comparing benchmark results against the known
objective values above. The 11 affected NETLIB problems (those with `FR` or
`MI` bounds) are: `capri`, `cycle`, `greenbeb`, `modszk1`, `pilot_ja`,
`pilot_we`, `perold`, `pilot4`, `tuff`, `stair`, `vtp_base`.

For most of these, Python solvers (CVXPY backends and `scipy_linprog` alike)
report `OPTIMAL`/`INFEASIBLE`/`UNBOUNDED` with objective values that do not
match the known value — e.g. `capri`'s known optimum is `2690.0129`, but
CVXPY backends report values from `-1668.57` to `-86.8`.

`scripts/data_loaders/matlab/mps_loader.m` (added alongside this check)
performs the reordering correctly and matches the known values closely for
10 of the 11 problems (`modszk1` remains unexplained and needs further
investigation).

**This bug has not been fixed** — fixing `mps_loader.py` (and verifying the
downstream CVXPY/SciPy bounds reconstruction) is a separate task.

## Related files

- [`config/problem_registry.yaml`](../../config/problem_registry.yaml) — the `known_objective_value` fields themselves
- [`scripts/data_loaders/python/mps_loader.py`](../../scripts/data_loaders/python/mps_loader.py) — Python MPS loader (has the ordering bug)
- [`scripts/data_loaders/matlab/mps_loader.m`](../../scripts/data_loaders/matlab/mps_loader.m) — MATLAB MPS loader (reorders correctly)
