"""
Memory-leak probe for the GODMAX pipeline.  READ-ONLY: touches no src code.

Goal: find WHERE memory grows across repeated use of the halo model
(as happens inside an inference / MCMC / Fisher loop).

Three prime suspects for JAX halo models:
  A) jit-cache growth: every method is @partial(jit, static_argnums=(0,)),
     so `self` is a STATIC arg.  A *new object instance each iteration*
     => a fresh XLA compilation cached forever, keyed on the new self.
     RSS climbs, compile count climbs, and it never plateaus.
  B) live-buffer growth: DeviceArrays held alive by lingering references
     (attributes stashed on long-lived objects, growing lists/dicts).
  C) tracer leaks: a traced value escaping a jit scope (caught by
     jax.check_tracer_leaks).

Run:
    python memory_leak_probe.py                 # all probes, default 25 iters
    python memory_leak_probe.py --iters 40
    python memory_leak_probe.py --stage cl      # base|profiles|pkz|cl|cov
    python memory_leak_probe.py --mode rebuild  # rebuild|reuse|both
    python memory_leak_probe.py --leakcheck     # enable jax tracer-leak check (slow)

Interpretation:
  - RSS + compile_count both rise ~linearly in `rebuild` mode but are FLAT
    in `reuse` mode  => suspect A (static-self jit cache). The fix lives in
    how the caller loops, or in making `self` hashable/stable, NOT here.
  - live_bytes rises in `reuse` mode too => suspect B (real buffer leak).
  - Non-zero tracer-leak report => suspect C.
"""

import argparse
import gc
import os
import sys

# Make `godmax` importable without an editable install (src layout).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
sys.path.insert(0, _HERE)  # for `tests.conftest`

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Config: mirror tests/conftest.py exactly (tiny grids, all physics on)
# ---------------------------------------------------------------------------
from tests.conftest import (  # noqa: E402
    SIM_PARAMS,
    HALO_PARAMS,
    ANALYSIS,
    OTHER_PARAMS,
)


# ---------------------------------------------------------------------------
# Memory / cache sampling helpers
# ---------------------------------------------------------------------------
def _rss_mb():
    """Resident set size in MB (psutil if present, else /proc)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e3  # kB -> MB
    return float("nan")


def _live_arrays():
    """(count, total_bytes) of arrays JAX currently holds on device."""
    try:
        arrs = jax.live_arrays()
    except Exception:
        return (-1, -1)
    total = 0
    for a in arrs:
        try:
            total += a.nbytes
        except Exception:
            pass
    return (len(arrs), total)


def _compile_count():
    """Total XLA compilations so far, via jax's global monitoring hook."""
    # jax exposes cumulative compile events through the internal counter.
    try:
        from jax._src import monitoring  # noqa

        # No public cumulative counter; fall back to per-fn cache sizes below.
    except Exception:
        pass
    return None


def _cache_sizes(objs):
    """Sum jit _cache_size() over the jitted methods reachable on given objs' classes."""
    total = 0
    seen = set()
    for obj in objs:
        if obj is None:
            continue
        cls = type(obj)
        for name in dir(cls):
            attr = getattr(cls, name, None)
            cs = getattr(attr, "_cache_size", None)
            if callable(cs) and id(attr) not in seen:
                seen.add(id(attr))
                try:
                    total += cs()
                except Exception:
                    pass
    return total


def sample(tag, i, objs):
    gc.collect()
    rss = _rss_mb()
    n_arr, n_bytes = _live_arrays()
    csize = _cache_sizes(objs)
    print(
        f"[{tag:8s}] iter {i:3d} | RSS {rss:8.1f} MB | "
        f"live_arrays {n_arr:6d} ({n_bytes/1e6:8.1f} MB) | "
        f"jit_cache_entries {csize:5d}"
    )
    return rss, n_arr, n_bytes, csize


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def build(stage):
    """Build the pipeline up to `stage` from scratch; return the last obj + all objs."""
    from godmax.base_class import base_class

    base = base_class(SIM_PARAMS, HALO_PARAMS, ANALYSIS, OTHER_PARAMS)
    objs = [base]
    if stage == "base":
        return base, objs

    from godmax.get_radial_profiles import Profiles

    prof = Profiles(SIM_PARAMS, HALO_PARAMS, ANALYSIS, OTHER_PARAMS, base_class_obj=base)
    objs.append(prof)
    if stage == "profiles":
        return prof, objs

    from godmax.get_Pkzs import get_Pkz

    pkz = get_Pkz(SIM_PARAMS, HALO_PARAMS, ANALYSIS, OTHER_PARAMS, Profiles_obj=prof)
    objs.append(pkz)
    if stage == "pkz":
        return pkz, objs

    from godmax.get_Cls import get_Cl

    cl = get_Cl(SIM_PARAMS, HALO_PARAMS, ANALYSIS, OTHER_PARAMS, Pkz_obj=pkz)
    objs.append(cl)
    if stage == "cl":
        return cl, objs

    from godmax.get_covs import get_cov

    cov = get_cov(SIM_PARAMS, HALO_PARAMS, ANALYSIS, OTHER_PARAMS, Cl_obj=cl)
    objs.append(cov)
    return cov, objs


# ---------------------------------------------------------------------------
# Probe modes
# ---------------------------------------------------------------------------
def probe_rebuild(stage, iters):
    """Rebuild the WHOLE pipeline every iteration (simulates naive loop).
    If suspect A is real: RSS + jit_cache_entries climb, never plateau."""
    print(f"\n=== REBUILD mode, stage={stage} (fresh objects each iter) ===")
    rows = []
    for i in range(iters):
        last, objs = build(stage)
        rows.append(sample("rebuild", i, objs))
        del last, objs  # drop refs; gc + jax should reclaim IF no leak
    return rows


def probe_reuse(stage, iters):
    """Build ONCE, then re-evaluate on the same object each iteration.
    Baseline: this SHOULD stay flat. If it climbs => real buffer leak (B)."""
    print(f"\n=== REUSE mode, stage={stage} (one object, re-eval each iter) ===")
    last, objs = build(stage)
    rows = []
    for i in range(iters):
        # Re-touch heavy attributes to force any lazy recompute paths.
        for name in dir(last):
            if name.startswith("Cl_") or name.startswith("P") and "mat" in name:
                a = getattr(last, name, None)
                if a is not None:
                    try:
                        _ = jax.numpy.asarray(a).sum().block_until_ready()
                    except Exception:
                        pass
        rows.append(sample("reuse", i, objs))
    return rows


def probe_leakcheck(stage):
    """Build once under jax tracer-leak checking. Reports suspect C."""
    print(f"\n=== TRACER-LEAK CHECK, stage={stage} ===")
    try:
        with jax.check_tracer_leaks():
            _last, _objs = build(stage)
        print("No tracer leak detected during build.")
    except Exception as e:
        print(f"TRACER LEAK / error during checked build:\n  {type(e).__name__}: {e}")


def summarize(tag, rows):
    if len(rows) < 2:
        return
    r0, rN = rows[1], rows[-1]  # skip iter0 (cold JIT)
    d_rss = rN[0] - r0[0]
    d_arr = rN[1] - r0[1]
    d_cache = rN[3] - r0[3]
    n = len(rows) - 1
    verdict = "LEAK-SUSPECT" if (d_rss > 20 or d_cache > 0 or d_arr > 50) else "flat/ok"
    print(
        f"\n>>> {tag} over {n} iters (iter1->last): "
        f"dRSS {d_rss:+.1f} MB, dlive_arrays {d_arr:+d}, "
        f"djit_cache {d_cache:+d}  ==> {verdict}"
    )


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument(
        "--stage", default="cl", choices=["base", "profiles", "pkz", "cl", "cov"]
    )
    ap.add_argument("--mode", default="both", choices=["rebuild", "reuse", "both"])
    ap.add_argument("--leakcheck", action="store_true")
    args = ap.parse_args()

    print(f"jax {jax.__version__} | devices {jax.devices()} | x64 on")
    print(f"stage={args.stage} iters={args.iters} mode={args.mode}")

    if args.leakcheck:
        probe_leakcheck(args.stage)

    if args.mode in ("reuse", "both"):
        rows = probe_reuse(args.stage, args.iters)
        summarize("REUSE  ", rows)

    if args.mode in ("rebuild", "both"):
        rows = probe_rebuild(args.stage, args.iters)
        summarize("REBUILD", rows)


if __name__ == "__main__":
    main()
