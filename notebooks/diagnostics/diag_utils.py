"""
Diagnostic utilities for GODMAX: data vector assembly, covariance matrix
construction, and n(z) conversion.

Probe conventions (matching Cl_result_dict / covtot_dict keys from get_cov):
    'yy'  — tSZ auto            bin keys: bin_0_0       (0-indexed, single entry)
    'ky'  — lensing × tSZ       bin keys: bin_{s}_0     (source bins 1-indexed, y=0)
    'kk'  — lensing auto        bin keys: bin_{s1}_{s2} (source bins 1-indexed)
    'gy'  — galaxy × tSZ        bin keys: bin_{l}_0     (lens bins 1-indexed, y=0)
    'gk'  — galaxy × lensing    bin keys: bin_{l}_{s}   (lens × source, 1-indexed)
    'gg'  — galaxy auto         bin keys: bin_{l1}_{l2} (lens bins 1-indexed)

Default data-vector bin selection:
    'kk'  → upper triangle (bin2 >= bin1)
    'gg'  → diagonal only  (bin2 == bin1)
    all others → every bin combination stored in Cl_result_dict
"""

import numpy as np
import matplotlib.pyplot as plt
import jax_cosmo.background as bkgrd
from jax_cosmo.background import radial_comoving_distance
from astropy import constants as const


# ---------------------------------------------------------------------------
# Probe metadata
# ---------------------------------------------------------------------------

#: LaTeX labels for the six standard probes.
PROBE_LATEX = {
    'yy': r'$\langle yy \rangle$',
    'ky': r'$\langle \kappa y \rangle$',
    'kk': r'$\langle \kappa\kappa \rangle$',
    'gy': r'$\langle gy \rangle$',
    'gk': r'$\langle g\kappa \rangle$',
    'gg': r'$\langle gg \rangle$',
}

#: Default bin-selection rules:
#:   'upper_tri' — bin2 >= bin1  (upper triangle, incl. diagonal)
#:   'diag'      — bin2 == bin1  (diagonal only)
#:   'all'       — every entry in bin_combs
DEFAULT_BIN_SELECTION = {
    'kk': 'upper_tri',
    'gg': 'diag',
    'ky': 'all',
    'gy': 'all',
    'gk': 'all',
    'yy': 'all',
}


# ---------------------------------------------------------------------------
# n(z) conversion
# ---------------------------------------------------------------------------

def to_comoving_nz(z_array, nz_array, cosmo):
    """Convert n(z) [gal/deg²/dz] to comoving number density [gal/(Mpc/h)³].

    Parameters
    ----------
    z_array : array_like, shape (nz,)
        Redshift grid.
    nz_array : array_like, shape (nz,)
        Galaxy counts per square degree per unit redshift.
    cosmo : jax_cosmo.Cosmology
        Cosmology object.

    Returns
    -------
    n_comoving : ndarray, shape (nz,)
        Comoving number density in (Mpc/h)^{-3}.
    """
    a_arr = 1.0 / (1.0 + np.asarray(z_array))
    chi = np.asarray(radial_comoving_distance(cosmo, a_arr))
    # H(a) from jax_cosmo [km/s / (Mpc/h)]; c [km/s] → dchi/dz [Mpc/h]
    dchi_dz = (const.c.value * 1e-3) / np.asarray(bkgrd.H(cosmo, a_arr))
    # dV = chi² dchi/dz dΩ; convert sr→deg²
    sr_per_deg2 = (np.pi / 180.) ** 2
    dV_dzdeg2 = chi ** 2 * dchi_dz * sr_per_deg2
    return np.asarray(nz_array) / dV_dzdeg2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _canonical_probe(probe, Cl_result_dict):
    """Return probe key as stored in Cl_result_dict; handles 'yk' → 'ky' etc."""
    if probe in Cl_result_dict:
        return probe
    rev = probe[::-1]
    if rev in Cl_result_dict:
        return rev
    raise KeyError(
        f"Probe '{probe}' (and its reverse '{rev}') not found in Cl_result_dict. "
        f"Available probes: {[k for k in Cl_result_dict if k not in ('l_array_survey', 'dl_array_survey')]}"
    )


def _passes_selection(b1, b2, rule):
    """Return True if the bin pair satisfies the selection rule."""
    if rule == 'upper_tri':
        return b2 >= b1
    if rule == 'diag':
        return b2 == b1
    return True  # 'all'


# ---------------------------------------------------------------------------
# Entry list
# ---------------------------------------------------------------------------

def get_data_vector_entries(Cl_result_dict, probes, bin_selection=None):
    """Build the ordered list of ``[probe, [bin1, bin2]]`` entries.

    Parameters
    ----------
    Cl_result_dict : dict
        ``get_cov.Cl_result_dict``.
    probes : list of str
        Probe names in the desired data-vector order, e.g.
        ``['gy', 'gg']`` or ``['ky', 'kk', 'gy', 'gg', 'gk']``.
    bin_selection : dict, optional
        Override the default selection rule for any probe.
        Keys are probe names, values are ``'upper_tri'``, ``'diag'``, or
        ``'all'``.  Unspecified probes fall back to ``DEFAULT_BIN_SELECTION``.

    Returns
    -------
    entries : list of [str, [int, int]]
        Ordered data-vector entries.  Each element is
        ``[probe_key, [bin1, bin2]]``.
    count_per_probe : list of int
        Number of entries contributed by each probe (same order as ``probes``).
    """
    sel = {**DEFAULT_BIN_SELECTION, **(bin_selection or {})}
    entries = []
    count_per_probe = []

    for probe in probes:
        key = _canonical_probe(probe, Cl_result_dict)
        rule = sel.get(key, 'all')
        count = 0
        for b1, b2 in Cl_result_dict[key]['bin_combs']:
            if _passes_selection(b1, b2, rule):
                entries.append([key, [b1, b2]])
                count += 1
        count_per_probe.append(count)

    return entries, count_per_probe


# ---------------------------------------------------------------------------
# Data vector
# ---------------------------------------------------------------------------

def build_data_vector(Cl_result_dict, probes,
                      key='tot_ellsurvey', bin_selection=None):
    """Build a flat 1-D data vector from ``Cl_result_dict``.

    Parameters
    ----------
    Cl_result_dict : dict
        ``get_cov.Cl_result_dict``.
    probes : list of str
        Probe names in the desired order.
    key : str, optional
        Which spectrum to use per entry:

        * ``'tot_ellsurvey'``          — signal only  (default)
        * ``'tot_plus_noise_ellsurvey'`` — signal + noise
        * ``'noise_ellsurvey'``        — noise only (only exists for kk diagonal and gg diagonal)
    bin_selection : dict, optional
        Passed to ``get_data_vector_entries``.

    Returns
    -------
    dv : ndarray, shape (n_entries × nell,)
        Flat concatenated data vector.
    entries : list of [str, [int, int]]
        The ordered entry list.
    nell : int
        Number of multipole bins.
    """
    entries, _ = get_data_vector_entries(Cl_result_dict, probes, bin_selection)
    nell = len(Cl_result_dict['l_array_survey'])

    blocks = []
    for probe, (b1, b2) in entries:
        bin_key = f'bin_{b1}_{b2}'
        entry = Cl_result_dict[probe][bin_key]
        if key not in entry:
            raise KeyError(
                f"Key '{key}' not found for probe '{probe}' bin_key '{bin_key}'. "
                f"Available keys: {list(entry.keys())}"
            )
        blocks.append(np.array(entry[key]))

    dv = np.concatenate(blocks) if blocks else np.array([])
    return dv, entries, nell


# ---------------------------------------------------------------------------
# Covariance matrix
# ---------------------------------------------------------------------------

def _lookup_cov_block(covtot_dict, probe1, b1, b2, probe2, b3, b4):
    """Return the (nell × nell) covariance block.

    Tries both probe orderings and both bin orderings.  The transpose is
    returned if the reversed ordering is found (ensuring Cov[i,j] = Cov[j,i].T).

    Raises ``KeyError`` if the block is absent from ``covtot_dict``.
    """
    key_fwd = f'{probe1}_{probe2}'
    key_rev = f'{probe2}_{probe1}'
    bkey_fwd = f'bin_{b1}_{b2}_{b3}_{b4}'
    bkey_rev = f'bin_{b3}_{b4}_{b1}_{b2}'

    if key_fwd in covtot_dict:
        d = covtot_dict[key_fwd]
        if bkey_fwd in d:
            return np.array(d[bkey_fwd]), False
    if key_rev in covtot_dict:
        d = covtot_dict[key_rev]
        if bkey_rev in d:
            return np.array(d[bkey_rev]).T, True

    raise KeyError(
        f"Block not found: ({probe1}, bin_{b1}_{b2}) × ({probe2}, bin_{b3}_{b4}). "
        f"Check that stats_for_cov covers both probes."
    )


def build_cov_matrix(covtot_dict, Cl_result_dict, probes,
                     bin_selection=None, symmetrize=True,
                     warn_missing=True):
    """Build the full joint covariance matrix for the requested probes.

    Off-diagonal probe-pair blocks are looked up in ``covtot_dict`` using
    both probe orderings automatically.  Blocks absent from ``covtot_dict``
    (i.e. probe pairs not in ``stats_for_cov``) are left as zero.

    Parameters
    ----------
    covtot_dict : dict
        ``get_cov.covtot_dict``.
    Cl_result_dict : dict
        ``get_cov.Cl_result_dict`` (used for bin lists and ell grid).
    probes : list of str
        Probe names in the desired data-vector order.
    bin_selection : dict, optional
        Passed to ``get_data_vector_entries``.
    symmetrize : bool, optional
        Copy the upper triangle to the lower triangle after assembly
        (default True).  Avoids floating-point asymmetry from block lookups.
    warn_missing : bool, optional
        Print a warning for each missing cross-block (default True).

    Returns
    -------
    cov : ndarray, shape (n_entries*nell, n_entries*nell)
        Full covariance matrix.
    entries : list of [str, [int, int]]
        Data-vector entry list.
    count_per_probe : list of int
        Number of ell-blocks contributed by each probe.
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
            except KeyError as exc:
                if warn_missing:
                    print(f"[build_cov_matrix] Missing block (left as zero): {exc}")

    if symmetrize:
        upper = np.triu(cov)
        cov = upper + upper.T - np.diag(np.diag(cov))

    return cov, entries, count_per_probe, nell


# ---------------------------------------------------------------------------
# Probe-boundary helpers
# ---------------------------------------------------------------------------

def get_probe_boundaries(count_per_probe, nell):
    """Compute pixel-space boundaries for a covariance matrix plot.

    Parameters
    ----------
    count_per_probe : list of int
        Number of bin-combos per probe (from ``get_data_vector_entries``).
    nell : int
        Number of multipole bins per ell block.

    Returns
    -------
    dict with keys:
        ``'edges'``    — boundary pixel indices, length n_probes + 1
        ``'centers'``  — centre pixel indices,   length n_probes
        ``'widths'``   — pixel widths,            length n_probes
    """
    widths  = np.array(count_per_probe, dtype=float) * nell
    edges   = np.concatenate([[0], np.cumsum(widths)])
    centers = edges[:-1] + widths / 2.
    return {'edges': edges, 'centers': centers, 'widths': widths}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cov_matrix(cov, count_per_probe, probes, nell,
                    ax=None, vmin=-100, vmax=-20,
                    line_color='white', line_lw=1.0,
                    probe_labels=None, colorbar=True,
                    **imshow_kw):
    """Plot log|covariance| with probe-boundary annotations.

    Parameters
    ----------
    cov : ndarray, shape (N, N)
        Full covariance matrix.
    count_per_probe : list of int
        Number of bin-combos per probe.
    probes : list of str
        Probe names in data-vector order.
    nell : int
        Number of multipole bins.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if None.
    vmin, vmax : float
        Colour-scale limits for ``log|cov|``.
    line_color : str
        Colour of probe-boundary lines.
    line_lw : float
        Width of probe-boundary lines.
    probe_labels : dict, optional
        Override the default LaTeX label for any probe.
    colorbar : bool, optional
        Draw a colour bar (default True).
    **imshow_kw
        Extra keyword arguments forwarded to ``ax.imshow``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    im : matplotlib.image.AxesImage
    """
    labels = {**PROBE_LATEX, **(probe_labels or {})}

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))

    im = ax.imshow(
        np.log(np.abs(cov) + 1e-70),
        origin='lower', vmin=vmin, vmax=vmax,
        **imshow_kw
    )

    bounds  = get_probe_boundaries(count_per_probe, nell)
    edges   = bounds['edges']
    centers = bounds['centers']

    # Lines between probes (skip outer edges at 0 and N)
    for edge in edges[1:-1]:
        ax.axhline(edge - 0.5, color=line_color, lw=line_lw)
        ax.axvline(edge - 0.5, color=line_color, lw=line_lw)

    tick_labels = [labels.get(p, p) for p in probes]
    ax.set_xticks(centers)
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.set_yticks(centers)
    ax.set_yticklabels(tick_labels, fontsize=13)
    ax.set_xlim(-0.5, cov.shape[1] - 0.5)
    ax.set_ylim(-0.5, cov.shape[0] - 0.5)

    if colorbar:
        plt.colorbar(im, ax=ax, label=r'$\log|\mathrm{Cov}|$',
                     fraction=0.046, pad=0.04)

    return ax, im


# ---------------------------------------------------------------------------
# Full pipeline convenience wrapper
# ---------------------------------------------------------------------------

def make_cov_matrix_and_plot(covtot_dict, Cl_result_dict, probes,
                              bin_selection=None,
                              plot=True, figsize=(10, 10),
                              vmin=-100, vmax=-20, **plot_kw):
    """Build covariance and optionally plot it in one call.

    Parameters
    ----------
    covtot_dict, Cl_result_dict, probes, bin_selection
        Forwarded to ``build_cov_matrix``.
    plot : bool
        Draw the matrix if True (default).
    figsize, vmin, vmax, **plot_kw
        Forwarded to ``plot_cov_matrix``.

    Returns
    -------
    cov : ndarray
    entries : list
    count_per_probe : list of int
    nell : int
    fig : matplotlib.figure.Figure or None
    """
    cov, entries, count_per_probe, nell = build_cov_matrix(
        covtot_dict, Cl_result_dict, probes, bin_selection
    )

    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        plot_cov_matrix(cov, count_per_probe, probes, nell,
                        ax=ax, vmin=vmin, vmax=vmax, **plot_kw)
        fig.tight_layout()

    return cov, entries, count_per_probe, nell, fig
