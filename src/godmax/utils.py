"""
Utility functions for building data vectors, covariance matrices, and n(z) conversions.

Probe conventions (matching Cl_result_dict / covtot_dict keys):
    'yy'  — tSZ auto            bin keys: bin_0_0       (single entry, 0-indexed)
    'ky'  — lensing × tSZ       bin keys: bin_{s}_0     (source bins 1-indexed, y=0)
    'kk'  — lensing auto        bin keys: bin_{s1}_{s2} (source bins 1-indexed)
    'gy'  — galaxy × tSZ        bin keys: bin_{l}_0     (lens bins 1-indexed, y=0)
    'gk'  — galaxy × lensing    bin keys: bin_{l}_{s}   (lens × source, 1-indexed)
    'gg'  — galaxy auto         bin keys: bin_{l1}_{l2} (lens bins 1-indexed)

Data-vector bin-selection defaults:
    'kk'  → upper triangle (bin2 >= bin1): exploit C_kk(b1,b2) = C_kk(b2,b1)
    'gg'  → diagonal only  (bin2 == bin1): auto-correlations per lens bin
    all others → all bin combinations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import jax_cosmo.background as bkgrd
from jax_cosmo.background import radial_comoving_distance
from astropy import constants as const


# ---------------------------------------------------------------------------
# Probe metadata
# ---------------------------------------------------------------------------

#: Default LaTeX label for each probe.
PROBE_LATEX = {
    'yy': r'$\langle yy \rangle$',
    'ky': r'$\langle \kappa y \rangle$',
    'kk': r'$\langle \kappa\kappa \rangle$',
    'gy': r'$\langle gy \rangle$',
    'gk': r'$\langle g\kappa \rangle$',
    'gg': r'$\langle gg \rangle$',
}

#: Default bin-selection rule per probe.
#:   'upper_tri' — include only pairs with bin2 >= bin1
#:   'diag'      — include only pairs with bin2 == bin1
#:   'all'       — include every bin combination stored in Cl_result_dict
DEFAULT_BIN_SELECTION = {
    'kk': 'upper_tri',
    'gg': 'diag',
    'ky': 'all',
    'gy': 'all',
    'gk': 'all',
    'yy': 'all',
}


# ---------------------------------------------------------------------------
# n(z) utilities
# ---------------------------------------------------------------------------

def to_comoving_nz(z_array, nz_array, cosmo):
    """Convert n(z) [gal/deg²/dz] to comoving number density [gal/(Mpc/h)³].

    Parameters
    ----------
    z_array : array_like, shape (nz,)
        Redshift values.
    nz_array : array_like, shape (nz,)
        Galaxy number counts per square degree per unit redshift.
    cosmo : jax_cosmo.Cosmology
        Cosmology object.

    Returns
    -------
    n_comoving : ndarray, shape (nz,)
        Comoving number density in (Mpc/h)^{-3}.
    """
    a_arr = 1.0 / (1.0 + np.asarray(z_array))
    chi = np.asarray(radial_comoving_distance(cosmo, a_arr))
    # H(a) from jax_cosmo is in units of (km/s)/(Mpc/h); c in km/s → dchi/dz in Mpc/h
    dchi_dz = (const.c.value * 1e-3) / np.asarray(bkgrd.H(cosmo, a_arr))
    # comoving volume element per deg² per dz:  χ² dχ/dz  [Mpc/h³ / deg²]
    sr_per_deg2 = (np.pi / 180.) ** 2
    dV_dzdeg2 = chi ** 2 * dchi_dz * sr_per_deg2
    return np.asarray(nz_array) / dV_dzdeg2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _canonical_probe(probe, Cl_result_dict):
    """Return the canonical probe key as it appears in Cl_result_dict.

    Handles reversed spellings (e.g. 'yk' → 'ky').
    """
    if probe in Cl_result_dict:
        return probe
    rev = probe[::-1]
    if rev in Cl_result_dict:
        return rev
    raise KeyError(f"Probe '{probe}' (or '{rev}') not found in Cl_result_dict.")


def _passes_selection(probe, b1, b2, rule):
    """Return True if the bin pair (b1, b2) satisfies the selection rule."""
    if rule == 'upper_tri':
        return b2 >= b1
    if rule == 'diag':
        return b2 == b1
    # 'all'
    return True


# ---------------------------------------------------------------------------
# Data-vector entry list
# ---------------------------------------------------------------------------

def get_data_vector_entries(Cl_result_dict, probes, bin_selection=None):
    """Build the ordered list of (probe, [bin1, bin2]) entries for a data vector.

    Parameters
    ----------
    Cl_result_dict : dict
        Output of ``get_cov.Cl_result_dict``.
    probes : list of str
        Ordered list of probe names, e.g. ``['gy', 'gg']`` or
        ``['ky', 'kk', 'gy', 'gg', 'gk']``.
    bin_selection : dict, optional
        Override the default bin-selection rule for any probe.
        Keys are probe names, values are ``'upper_tri'``, ``'diag'``, or ``'all'``.
        Unspecified probes fall back to ``DEFAULT_BIN_SELECTION``.

    Returns
    -------
    entries : list of [str, [int, int]]
        Each element is ``[probe, [bin1, bin2]]`` in data-vector order.
    count_per_probe : list of int
        Number of entries contributed by each probe (same order as ``probes``).
    """
    sel = {**DEFAULT_BIN_SELECTION, **(bin_selection or {})}

    entries = []
    count_per_probe = []

    for probe in probes:
        probe_key = _canonical_probe(probe, Cl_result_dict)
        rule = sel.get(probe_key, 'all')
        count = 0
        for b1, b2 in Cl_result_dict[probe_key]['bin_combs']:
            if _passes_selection(probe_key, b1, b2, rule):
                entries.append([probe_key, [b1, b2]])
                count += 1
        count_per_probe.append(count)

    return entries, count_per_probe


# ---------------------------------------------------------------------------
# Data vector
# ---------------------------------------------------------------------------

def build_data_vector(Cl_result_dict, probes, key='tot_ellsurvey',
                      bin_selection=None):
    """Build a flat 1-D data vector from ``Cl_result_dict``.

    Parameters
    ----------
    Cl_result_dict : dict
        Output of ``get_cov.Cl_result_dict``.
    probes : list of str
        Ordered probe names.
    key : str, optional
        Which Cl entry to use.  Choices:
        ``'tot_ellsurvey'`` (signal only),
        ``'tot_plus_noise_ellsurvey'`` (signal + noise),
        ``'noise_ellsurvey'`` (noise only).
    bin_selection : dict, optional
        Passed to ``get_data_vector_entries``.

    Returns
    -------
    dv : ndarray, shape (n_entries * nell,)
        Flat data vector.
    entries : list of [str, [int, int]]
        Entry list (same order as ``dv`` blocks).
    nell : int
        Number of multipole bins.
    """
    entries, _ = get_data_vector_entries(Cl_result_dict, probes, bin_selection)
    nell = len(Cl_result_dict['l_array_survey'])

    blocks = []
    for probe, (b1, b2) in entries:
        bin_key = f'bin_{b1}_{b2}'
        block = np.array(Cl_result_dict[probe][bin_key][key])
        blocks.append(block)

    dv = np.concatenate(blocks) if blocks else np.array([])
    return dv, entries, nell


# ---------------------------------------------------------------------------
# Covariance matrix
# ---------------------------------------------------------------------------

def _lookup_cov_block(covtot_dict, probe1, b1, b2, probe2, b3, b4):
    """Retrieve a covariance block, trying both probe orderings.

    Returns (block as ndarray, was_transposed).
    Raises KeyError if the block cannot be found.
    """
    key_fwd = f'{probe1}_{probe2}'
    key_rev = f'{probe2}_{probe1}'
    bkey_fwd = f'bin_{b1}_{b2}_{b3}_{b4}'
    bkey_rev = f'bin_{b3}_{b4}_{b1}_{b2}'

    if key_fwd in covtot_dict and bkey_fwd in covtot_dict[key_fwd]:
        return np.array(covtot_dict[key_fwd][bkey_fwd]), False
    if key_rev in covtot_dict and bkey_rev in covtot_dict[key_rev]:
        return np.array(covtot_dict[key_rev][bkey_rev]).T, True
    raise KeyError(
        f"Covariance block not found for "
        f"({probe1}, bin_{b1}_{b2}) × ({probe2}, bin_{b3}_{b4})."
    )


def build_cov_matrix(covtot_dict, Cl_result_dict, probes,
                     bin_selection=None, symmetrize=True):
    """Build the full joint covariance matrix for the requested probes.

    Off-diagonal probe-pair blocks are looked up in ``covtot_dict`` with
    automatic handling of both probe orderings (e.g. 'ky_kk' vs 'kk_ky').
    Missing blocks are left as zero (with a warning printed).

    Parameters
    ----------
    covtot_dict : dict
        Output of ``get_cov.covtot_dict``.
    Cl_result_dict : dict
        Output of ``get_cov.Cl_result_dict`` (needed for bin lists).
    probes : list of str
        Ordered probe names.
    bin_selection : dict, optional
        Passed to ``get_data_vector_entries``.
    symmetrize : bool, optional
        If True (default), enforce exact symmetry by copying the upper
        triangle to the lower triangle after assembly.

    Returns
    -------
    cov : ndarray, shape (n_entries*nell, n_entries*nell)
        Full covariance matrix.
    entries : list of [str, [int, int]]
        Data-vector entry list.
    count_per_probe : list of int
        Number of ℓ-block rows/columns per probe.
    nell : int
        Number of multipole bins.
    """
    entries, count_per_probe = get_data_vector_entries(
        Cl_result_dict, probes, bin_selection
    )
    nell = len(Cl_result_dict['l_array_survey'])
    n = len(entries)
    cov = np.zeros((n * nell, n * nell))

    for jp1, (probe1, (b1, b2)) in enumerate(entries):
        for jp2, (probe2, (b3, b4)) in enumerate(entries):
            try:
                block, _ = _lookup_cov_block(
                    covtot_dict, probe1, b1, b2, probe2, b3, b4
                )
                cov[jp1*nell:(jp1+1)*nell, jp2*nell:(jp2+1)*nell] = block
            except KeyError:
                pass  # leave as zero; off-diagonal pairs not in stats_for_cov

    if symmetrize:
        upper = np.triu(cov)
        cov = upper + upper.T - np.diag(np.diag(cov))

    return cov, entries, count_per_probe, nell


# ---------------------------------------------------------------------------
# Probe-boundary helpers
# ---------------------------------------------------------------------------

def get_probe_boundaries(count_per_probe, nell):
    """Compute pixel-space probe boundaries for a covariance matrix plot.

    Parameters
    ----------
    count_per_probe : list of int
        Number of bin-combos per probe (from ``get_data_vector_entries``).
    nell : int
        Number of multipole bins.

    Returns
    -------
    dict with keys:
        'edges'   — array of boundary pixel indices (length n_probes + 1)
        'centers' — array of center pixel indices    (length n_probes)
        'widths'  — array of pixel widths            (length n_probes)
    """
    widths = np.array(count_per_probe) * nell
    edges = np.concatenate([[0], np.cumsum(widths)])
    centers = edges[:-1] + widths / 2.
    return {'edges': edges, 'centers': centers, 'widths': widths}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cov_matrix(cov, count_per_probe, probes, nell,
                    ax=None, vmin=-100, vmax=-20,
                    line_color='white', line_lw=1.0,
                    probe_labels=None, **imshow_kw):
    """Plot log|covariance| as a colour map with probe-boundary annotations.

    Parameters
    ----------
    cov : ndarray, shape (N, N)
        Full covariance matrix (as returned by ``build_cov_matrix``).
    count_per_probe : list of int
        Number of bin-combos per probe.
    probes : list of str
        Probe names in data-vector order (used for axis tick labels).
    nell : int
        Number of multipole bins.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; created if None.
    vmin, vmax : float
        Colour-scale limits for ``log|cov|``.
    line_color : str
        Colour of probe-boundary lines.
    line_lw : float
        Line width of probe-boundary lines.
    probe_labels : dict, optional
        Override LaTeX label for any probe; falls back to ``PROBE_LATEX``.
    **imshow_kw
        Extra keyword arguments forwarded to ``ax.imshow``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    im : matplotlib.image.AxesImage
    """
    labels = {**PROBE_LATEX, **(probe_labels or {})}

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        np.log(np.abs(cov) + 1e-70),
        origin='lower', vmin=vmin, vmax=vmax,
        **imshow_kw
    )

    bounds = get_probe_boundaries(count_per_probe, nell)
    edges   = bounds['edges']
    centers = bounds['centers']

    # Draw dividing lines between probes (skip the outermost edges)
    for edge in edges[1:-1]:
        ax.axhline(edge - 0.5, color=line_color, lw=line_lw)
        ax.axvline(edge - 0.5, color=line_color, lw=line_lw)

    # Tick labels at probe centres
    tick_labels = [labels.get(p, p) for p in probes]
    ax.set_xticks(centers)
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.set_yticks(centers)
    ax.set_yticklabels(tick_labels, fontsize=13)

    ax.set_xlim(-0.5, cov.shape[1] - 0.5)
    ax.set_ylim(-0.5, cov.shape[0] - 0.5)

    plt.colorbar(im, ax=ax, label=r'$\log|\mathrm{Cov}|$', fraction=0.046, pad=0.04)
    return ax, im


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------

def make_cov_matrix_and_plot(covtot_dict, Cl_result_dict, probes,
                              bin_selection=None, plot=True,
                              figsize=(10, 10), vmin=-100, vmax=-20,
                              **plot_kw):
    """Build the covariance matrix and optionally plot it.

    Parameters
    ----------
    covtot_dict, Cl_result_dict, probes, bin_selection
        Forwarded to ``build_cov_matrix``.
    plot : bool
        If True, call ``plot_cov_matrix`` and return the figure.
    figsize, vmin, vmax, **plot_kw
        Forwarded to ``plot_cov_matrix``.

    Returns
    -------
    cov : ndarray
    entries : list
    count_per_probe : list
    nell : int
    fig : matplotlib.figure.Figure or None
    """
    cov, entries, count_per_probe, nell = build_cov_matrix(
        covtot_dict, Cl_result_dict, probes, bin_selection
    )
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        plot_cov_matrix(cov, count_per_probe, probes, nell, ax=ax,
                        vmin=vmin, vmax=vmax, **plot_kw)
        fig.tight_layout()

    return cov, entries, count_per_probe, nell, fig
