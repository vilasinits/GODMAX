"""Generate GODMAX status slides as a minimal white/black PPTX."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

BLACK  = RGBColor(0, 0, 0)
WHITE  = RGBColor(255, 255, 255)
GREY   = RGBColor(180, 180, 180)
LGREY  = RGBColor(240, 240, 240)
BLUE   = RGBColor(30, 80, 160)   # used sparingly for headings only

W = Inches(13.33)   # widescreen 16:9
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

blank_layout = prs.slide_layouts[6]   # completely blank

# ── helpers ──────────────────────────────────────────────────────────────────

def add_slide():
    return prs.slides.add_slide(blank_layout)

def rect(slide, left, top, width, height, fill=WHITE, line=False):
    shape = slide.shapes.add_shape(
        1,   # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line:
        shape.line.color.rgb = GREY
        shape.line.width = Pt(0.5)
    else:
        shape.line.fill.background()
    return shape

def textbox(slide, text, left, top, width, height,
            size=18, bold=False, color=BLACK, align=PP_ALIGN.LEFT,
            wrap=True):
    txb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Helvetica Neue"
    return txb

def title_slide_box(slide, title, subtitle=""):
    # thin top bar
    rect(slide, 0, 0, 13.33, 0.08, fill=BLUE)
    rect(slide, 0, 0.08, 13.33, 7.42, fill=WHITE)
    textbox(slide, title,    0.6, 2.6, 12.0, 1.2, size=36, bold=True,
            color=BLUE, align=PP_ALIGN.CENTER)
    if subtitle:
        textbox(slide, subtitle, 0.6, 3.9, 12.0, 0.8, size=20,
                color=BLACK, align=PP_ALIGN.CENTER)
    # thin bottom bar
    rect(slide, 0, 7.3, 13.33, 0.2, fill=BLUE)

def content_slide(slide, heading, bullets, note=""):
    """heading bar + bullet list."""
    rect(slide, 0, 0, 13.33, 0.08, fill=BLUE)
    rect(slide, 0, 0.08, 13.33, 7.42, fill=WHITE)
    # heading
    rect(slide, 0, 0.08, 13.33, 0.72, fill=BLUE)
    textbox(slide, heading, 0.3, 0.12, 12.7, 0.65,
            size=24, bold=True, color=WHITE)
    # bullets
    y = 1.05
    for b in bullets:
        indent = b.startswith("    ")
        text   = b.lstrip()
        prefix = "  •  " if not indent else "      –  "
        sz     = 17 if not indent else 15
        textbox(slide, prefix + text, 0.3, y, 12.5, 0.42,
                size=sz, color=BLACK)
        y += 0.38 if not indent else 0.33
    if note:
        rect(slide, 0, 7.1, 13.33, 0.2, fill=GREY)
        textbox(slide, note, 0.3, 7.12, 12.7, 0.2,
                size=11, color=BLACK)
    rect(slide, 0, 7.3, 13.33, 0.2, fill=BLUE)

def table_slide(slide, heading, col_headers, rows, col_widths=None):
    rect(slide, 0, 0, 13.33, 0.08, fill=BLUE)
    rect(slide, 0, 0.08, 13.33, 7.42, fill=WHITE)
    rect(slide, 0, 0.08, 13.33, 0.72, fill=BLUE)
    textbox(slide, heading, 0.3, 0.12, 12.7, 0.65,
            size=24, bold=True, color=WHITE)

    ncols = len(col_headers)
    if col_widths is None:
        col_widths = [13.0 / ncols] * ncols

    # header row
    x = 0.15
    for i, h in enumerate(col_headers):
        r = rect(slide, x, 1.05, col_widths[i]-0.05, 0.40, fill=GREY, line=True)
        textbox(slide, h, x+0.05, 1.07, col_widths[i]-0.15, 0.35,
                size=14, bold=True, color=BLACK)
        x += col_widths[i]

    # data rows
    for ri, row in enumerate(rows):
        y = 1.50 + ri * 0.42
        bg = WHITE if ri % 2 == 0 else RGBColor(245, 245, 245)
        x = 0.15
        for ci, cell in enumerate(row):
            rect(slide, x, y, col_widths[ci]-0.05, 0.38, fill=bg, line=True)
            textbox(slide, cell, x+0.05, y+0.02, col_widths[ci]-0.15, 0.34,
                    size=13, color=BLACK)
            x += col_widths[ci]

    rect(slide, 0, 7.3, 13.33, 0.2, fill=BLUE)

def equation_slide(slide, heading, equations, params):
    """Heading bar + equations (left 2/3) + key parameters box (right 1/3)."""
    rect(slide, 0, 0, 13.33, 0.08, fill=BLUE)
    rect(slide, 0, 0.08, 13.33, 7.42, fill=WHITE)
    rect(slide, 0, 0.08, 13.33, 0.72, fill=BLUE)
    textbox(slide, heading, 0.3, 0.12, 12.7, 0.65,
            size=22, bold=True, color=WHITE)

    # Left column: equations
    y = 1.02
    for eq in equations:
        if eq == "":
            y += 0.18          # blank-line spacer
            continue
        indented = eq.startswith("    ") or eq.startswith("\t")
        sz = 13 if indented else 14
        col = RGBColor(60, 60, 60) if indented else BLACK
        textbox(slide, eq, 0.35, y, 8.55, 0.42, size=sz, color=col)
        y += 0.40

    # Right column: parameters box
    box_top = 1.00
    box_h   = 6.10
    rect(slide, 9.2, box_top, 3.9, box_h, fill=LGREY, line=True)
    textbox(slide, "Key parameters", 9.35, box_top + 0.08, 3.60, 0.40,
            size=14, bold=True, color=BLUE)
    rect(slide, 9.2, box_top + 0.52, 3.9, 0.025, fill=GREY)   # separator
    yp = box_top + 0.60
    for p in params:
        textbox(slide, "  \u2022  " + p, 9.32, yp, 3.65, 0.42, size=12, color=BLACK)
        yp += 0.39

    rect(slide, 0, 7.3, 13.33, 0.2, fill=BLUE)


# ── Slide 1: Title ────────────────────────────────────────────────────────────
sl = add_slide()
title_slide_box(sl,
    "GODMAX: Theory Model & Code Status",
    "tSZ × Euclid Likelihood  |  Consortium Meeting Preparation")

# ── Slide 2: What GODMAX does ─────────────────────────────────────────────────
sl = add_slide()
content_slide(sl, "What GODMAX does", [
    "JAX-native differentiable halo model for cosmological observables",
    "Inputs: cosmology + astrophysical/HOD params",
    "Outputs: angular power spectra C(ℓ) + covariance matrix for all probe combinations",
    "Probes: κκ, κy, gy, gg, gk, yy",
    "    κ = CMB lensing convergence (single source plane)",
    "    y = Compton-y tSZ",
    "    g = galaxy overdensity (Euclid photometric)",
    "Designed to interface with CLOE (Euclid likelihood code)",
    "Enables NUTS sampling via NumPyro (differentiable C(ℓ) → gradients for free)",
])

# ── Slide 3: Pipeline ─────────────────────────────────────────────────────────
sl = add_slide()
content_slide(sl, "Theory pipeline", [
    "Halo mass function: Tinker 2008/2010",
    "Concentration: Duffy08 / Prada12 / Diemer15",
    "Matter profile: NFW + baryonification (BCMP, Schneider+19 / Pandey+24)",
    "    Gas params: θ_ej, θ_co, β evolving with M, z, concentration",
    "Pressure profile: HSE + non-thermal support  (α_nt, β_nt, n_nt)",
    "Galaxy profile: HOD — Leauthaud+11 SHMR  (N_cen, N_sat)",
    "3D power spectra P(k,z): 1-halo + 2-halo via FFTlog (mcfitjax)",
    "Baryonic suppression: S(k,z) = P_halofit / P_NFW applied to Pgm, Pgg",
    "C(ℓ): Limber approximation, beam suppression B(ℓ) for y-probes",
    "Covariance: Gaussian (Knox) + Non-Gaussian (1-halo trispectrum)",
])

# ── Slide 3a: Halo Mass Function ──────────────────────────────────────────────
sl = add_slide()
equation_slide(sl, "Theory: Halo Mass Function — Tinker 2008 / 2010",
    equations=[
        "dn/dM(M,z)  =  f(σ) · (ρ̄ₘ / M) · |d ln σ⁻¹ / dM|",
        "",
        "Tinker 2008 (abundance):  f(σ) = A · [(σ/b)⁻ᵃ + 1] · exp(−c/σ²)",
        "",
        "Tinker 2010 (bias):  separate calibration for large-scale bias b(M,z)",
        "    b(M,z) = 1 + [q ν² − 1]/δc + (2p/δc) / [1 + (q ν²)^p]",
        "",
        "Variance:  σ²(M,z) = ∫ P_lin(k,z) |W̃(kR)|² k² dk / 2π²",
        "    R = (3M / 4π ρ̄ₘ)^(1/3),   ν = δc / σ(M,z)",
    ],
    params=[
        "A, a, b, c — fitted to N-body",
        "Δ = 200 (w.r.t. ρ_m)",
        "δc = 1.686 (collapse threshold)",
        "hmf_model: T08 or T10",
        "σ8 — P_lin amplitude",
        "Ωm — sets ρ̄ₘ",
        "nM — mass grid resolution",
        "nz — redshift grid resolution",
    ],
)

# ── Slide 3b: Concentration–mass ─────────────────────────────────────────────
sl = add_slide()
equation_slide(sl, "Theory: Concentration–Mass Relation",
    equations=[
        "c(M,z)  =  A · (M / M_pivot)^B · (1 + z)^C",
        "",
        "Duffy+08:    A = 10.14,  B = −0.081,  C = −1.01",
        "    M_pivot = 2×10¹² M☉/h    (NFW fits to WMAP5 halos)",
        "",
        "Prada+12:    c inferred by matching σ(M,z) inversion",
        "    accounts for secondary halo properties (spin, shape)",
        "",
        "Diemer+15:   c(M,z) from peak height ν + accretion rate Γ",
        "    captures both relaxed and unrelaxed halo populations",
    ],
    params=[
        "conc_model: Duffy08,",
        "  Prada12, or Diemer15",
        "A, B, C — model-dependent",
        "M_pivot = 2×10¹² M☉/h",
        "r_s = r_vir / c(M,z)",
        "(used by NFW + BCMP",
        " and pressure profiles)",
    ],
)

# ── Slide 3c: NFW + BCMP baryonification ─────────────────────────────────────
sl = add_slide()
equation_slide(sl, "Theory: Matter Profile — NFW + Baryonification (BCMP)",
    equations=[
        "NFW:    ρ_NFW(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]",
        "    r_s = r_vir / c(M,z),   ρ_s set by enclosed mass M_vir",
        "",
        "BCMP gas profile (Schneider+19 / Pandey+24):",
        "    ρ_gas(r; θ_ej, θ_co, β, M, z, c)  =  M_gas · g(r)",
        "    g(r) ∝ [1 + (r/θ_co r_vir)²]^(−β/2) · exp[−(r/θ_ej r_vir)²]",
        "",
        "Total:  ρ_tot = ρ_DM + ρ_gas + ρ_stars   (CLM profile)",
        "    Backreaction: iterative ζ (adiabatic contraction) for CLM",
    ],
    params=[
        "θ_ej — ejection radius",
        "    (units of r_vir)",
        "θ_co — core radius",
        "    (units of r_vir)",
        "β — gas profile slope",
        "All evolve with M, z, c",
        "backreaction: True/False",
        "nr — radial grid points",
    ],
)

# ── Slide 3d: Pressure profile ────────────────────────────────────────────────
sl = add_slide()
equation_slide(sl, "Theory: Pressure Profile — GNFW + Non-thermal Support",
    equations=[
        "Generalised NFW (GNFW):",
        "    P_e(r) = P_0 · p(x),    x = r / r_500",
        "    p(x) = (x/xc)^(−γ) · [1 + (x/xc)^α]^((γ−δ)/α)",
        "",
        "Non-thermal pressure fraction:",
        "    f_nt(r) = α_nt · (r / r_500)^β_nt · (1 + z)^n_nt",
        "",
        "Effective thermal pressure:  P_th = P_tot · (1 − f_nt)",
        "",
        "Compton-y:  y = (σ_T / m_e c²) ∫ P_e dl   (l.o.s. integral)",
    ],
    params=[
        "α_nt — f_nt amplitude",
        "β_nt — radial slope",
        "n_nt — redshift evolution",
        "P_0 — normalisation",
        "α, γ, δ — GNFW shape",
        "    (fixed, literature vals)",
        "xc — core radius ratio",
        "HSE default: f_nt = 0",
    ],
)

# ── Slide 3e: HOD / SHMR ──────────────────────────────────────────────────────
sl = add_slide()
equation_slide(sl, "Theory: Galaxy Profile — HOD (Leauthaud+11 SHMR)",
    equations=[
        "Central occupation:",
        "    N_cen(M) = ½ · erfc[ log₁₀(M₁/M) / (√2 · σ_logM) ]",
        "",
        "Satellite occupation:",
        "    N_sat(M) = N_cen(M) · (M / M_sat)^α_sat · exp(−M_cut / M)",
        "",
        "Total HOD:  ⟨N(M)⟩ = N_cen(M) + N_sat(M)",
        "",
        "Galaxy kernel:   W_g(χ) = b_g · n(z) · H(z) / c",
        "    SHMR: M_star–M_halo from abundance matching",
    ],
    params=[
        "M₁ — char. halo mass",
        "σ_logM — mass scatter",
        "M_sat — satellite scale",
        "α_sat — satellite slope",
        "M_cut — satellite cutoff",
        "b_g — large-scale bias",
        "    (Tinker 2010)",
        "nbins_lens — n(z) bins",
    ],
)

# ── Slide 3f: 3D power spectra ────────────────────────────────────────────────
sl = add_slide()
equation_slide(sl, "Theory: 3D Power Spectra — 1-halo + 2-halo",
    equations=[
        "P^{AB}(k,z)  =  P^{AB}_{1h}(k,z)  +  P^{AB}_{2h}(k,z)",
        "",
        "1-halo:    P_{1h}^{AB}(k,z) = ∫ n(M) ũ_A(k|M,z) ũ_B(k|M,z) dM",
        "    ũ_A = Fourier transform of radial profile for field A",
        "",
        "2-halo:    P_{2h}^{AB}(k,z) = b_A(k,z) · b_B(k,z) · P_lin(k,z)",
        "",
        "Probes: mm, gm, gg, ym, yy, gy  (all pairs of κ,g,y fields)",
        "Evaluated via FFTlog (mcfitjax):  ξ(r) ↔ P(k)",
    ],
    params=[
        "nM — mass grid points",
        "nk — wavenumber points",
        "P_lin from jax_cosmo",
        "b_A — Tinker 2010 bias",
        "lowpass_Pmm1h_lowk:",
        "  suppress 1h at low k",
        "  (kthresh, default off)",
        "Probes: mm,gm,gg,ym,yy,gy",
    ],
)

# ── Slide 3g: Baryonic suppression ────────────────────────────────────────────
sl = add_slide()
equation_slide(sl, "Theory: Baryonic Suppression S(k,z)",
    equations=[
        "Suppression ratio:",
        "    S(k,z)  =  P_halofit(k,z)  /  P_NFW(k,z)",
        "",
        "Applied to matter-coupled probes (full S):",
        "    P_gg(k,z)  →  P_gg · S(k,z)",
        "    P_gm(k,z)  →  P_gm · S(k,z)",
        "",
        "Cross-spectrum correction — Fix C (√S prescription):",
        "    C^{gy}_ℓ  →  C^{gy}_ℓ · √S(ℓ/χ̄, z̄)",
        "    C^{κy}_ℓ  →  C^{κy}_ℓ · √S(ℓ/χ̄, z̄)",
        "    C^{yy}_ℓ  unchanged  (BCMP already captures baryons)",
    ],
    params=[
        "S(k,z) — derived ratio",
        "P_halofit — jax_cosmo",
        "P_NFW — halo model",
        "√S to gy, κy only",
        "(DMD congruence form",
        " → preserves PSD)",
        "Fix C not yet merged",
        "  into get_Cls.py",
    ],
)

# ── Slide 3h: C(ℓ) — Limber ──────────────────────────────────────────────────
sl = add_slide()
equation_slide(sl, "Theory: Angular Power Spectra — Limber Approximation",
    equations=[
        "C^{AB}_ℓ  =  ∫  W_A(χ) W_B(χ) / χ²  ·  P^{AB}((ℓ+½)/χ, z)  dχ",
        "",
        "Lensing kernel:   W_κ(χ) = (3H₀²Ωm/2c²) · χ(1+z) · (χ_s−χ)/χ_s",
        "tSZ kernel:       W_y(χ)  = 1 / (1+z)",
        "Galaxy kernel:    W_g(χ)  = b_g · n(z) · H(z) / c",
        "",
        "Beam suppression (ACT/Planck):",
        "    B_ℓ = exp[−ℓ(ℓ+1) σ²_beam / 2],   σ_beam = θ_FWHM/(2√(2 ln 2))",
        "    C^{yy}_ℓ × B²_ℓ,   C^{κy/gy}_ℓ × B¹_ℓ",
    ],
    params=[
        "nell — ℓ grid points",
        "nz — redshift grid",
        "θ_FWHM — beam FWHM",
        "α_ky, α_gy — beam power",
        "nbin_lens — lens bins",
        "nbin_source — source bins",
        "n(z) — photo-z dist.",
        "A_IA, η_IA — NLA params",
    ],
)

# ── Slide 3i: Covariance ──────────────────────────────────────────────────────
sl = add_slide()
equation_slide(sl, "Theory: Covariance — Gaussian (Knox) + Non-Gaussian (Trispectrum)",
    equations=[
        "Gaussian (Knox formula):",
        "    Cov^G_{AB,CD}(ℓ)  =  (C̃_{AC} C̃_{BD} + C̃_{AD} C̃_{BC})",
        "                          / [(2ℓ+1) Δℓ f_sky]",
        "    C̃_{AB}(ℓ) = C^{AB}_ℓ + N^{AB}_ℓ   (signal + noise)",
        "",
        "Non-Gaussian (1-halo trispectrum):",
        "    Cov^{NG}_{AB,CD}(ℓ₁,ℓ₂)  =  T^{AB,CD}(ℓ₁,ℓ₂) / (4π f_sky)",
        "    T^{AB,CD} = ∫∫ ũ_A(ℓ₁)ũ_B(ℓ₁) ũ_C(ℓ₂)ũ_D(ℓ₂) n(M) dM dz",
    ],
    params=[
        "f_sky — sky fraction",
        "Δℓ — bandpower width",
        "N^{AB}_ℓ — noise power",
        "stats_for_cov: probes",
        "Regularisation needed:",
        "  diag-shift ×1.1",
        "  → χ²/ndof ≈ 0.94",
        "(clip λ<0 leaves",
        "  matrix singular)",
    ],
)

# ── Slide 4: Probe status ─────────────────────────────────────────────────────
sl = add_slide()
table_slide(sl, "Probe status",
    ["Probe", "Theory ingredient", "Status"],
    [
        ["κκ",  "CMB lensing kernel W_κ, baryonified matter P(k,z)",   "✅  Working"],
        ["yy",  "GNFW pressure profile, Compton-y, beam²",              "✅  Working"],
        ["κy",  "Cross: CMB lensing × tSZ, beam¹",                     "✅  Working"],
        ["gy",  "HOD galaxy kernel × pressure profile, beam¹",         "✅  Working"],
        ["gg",  "HOD auto (pair counts), baryonic suppression",        "✅  Working"],
        ["gk",  "Galaxy × CMB lensing cross",                          "✅  Working"],
    ],
    col_widths=[1.4, 7.6, 4.0],
)

# ── Slide 5: Inputs needed ─────────────────────────────────────────────────────
sl = add_slide()
table_slide(sl, "Inputs needed",
    ["Input", "Source / Owner", "Status"],
    [
        ["Cosmological params (H0, Ωm, σ8, ns, Ob)",   "CLOE / Euclid fiducial",        "✅  Available"],
        ["tSZ beam FWHM",                               "ACT / Planck map specs",         "✅  Available"],
        ["C^yy_ℓ total + noise",                        "ACT / Planck",                   "⚠️  Need files"],
        ["Lens galaxy n(z)  [gy, gg, gk]",             "Euclid PhotoZ SWG",              "⚠️  Need DR1"],
        ["Source galaxy n(z)  [κκ, κy, gk]",           "Euclid WL SWG",                  "⚠️  Need DR1"],
        ["HOD calibration (N_cen, N_sat vs sims)",      "Galaxy clustering SWG",          "❌  Missing"],
        ["f_sky  (ACT × Euclid footprint overlap)",     "Maps / WL SWG",                  "⚠️  Need mask"],
        ["Non-thermal pressure priors (α_nt, β_nt)",   "tSZ / ICM SWG",                  "⚠️  Need priors"],
    ],
    col_widths=[5.0, 4.5, 3.5],
)

# ── Slide 6: What works ────────────────────────────────────────────────────────
sl = add_slide()
content_slide(sl, "Code status: what works", [
    "Full C(ℓ) pipeline for all 6 probes — JAX, differentiable, GPU-ready",
    "Gaussian covariance (Knox formula)",
    "Non-Gaussian covariance (1-halo trispectrum) — ℓ-index bug fixed",
    "CMB lensing kernel (single source plane at z ~ 1100)",
    "Baryonification suppression S(k,z) in all matter-coupled probes",
    "CLOE-compatible data-vector format (build_data_vector, build_bin_combinations)",
    "Linearised NUTS sampling via NumPyro — tested on single probes",
    "Regularisation strategies benchmarked — diag-shift ×1.1 gives χ²/ndof ≈ 1",
])

# ── Slide 7: Open issues ───────────────────────────────────────────────────────
sl = add_slide()
table_slide(sl, "Open issues",
    ["Issue", "Severity", "Notes / Fix"],
    [
        ["Gaussian cov non-PSD (P_sup asymmetry in gy, κy)",
         "Medium", "Fix C identified: apply √S to C^gy, C^κy — not merged yet"],
        ["HOD not validated vs simulations",
         "High",   "Need calibration from galaxy SWG"],
        ["Non-thermal pressure params unconstrained",
         "Medium", "Need priors from tSZ / ICM SWG"],
        ["No pseudo-Cℓ / mode-coupling (NaMaster)",
         "High",   "Analytic cov used for now; NaMaster integration pending"],
        ["Photo-z / shear calibration (Δz, m-bias)",
         "Low",    "Placeholder values; need Euclid calibration"],
        ["Multi-probe NUTS not yet validated",
         "Medium", "Single-probe works; multi-probe covariance regularisation needed"],
    ],
    col_widths=[5.5, 2.0, 5.5],
)

# ── Slide 8: SWG contacts ──────────────────────────────────────────────────────
sl = add_slide()
table_slide(sl, "SWG contacts needed before Consortium meeting",
    ["Topic", "SWG", "Ask"],
    [
        ["HOD calibration (N_cen, N_sat)",         "Galaxy clustering SWG",  "Calibrated HOD at Euclid cosmology + mass range"],
        ["Lens n(z) for gy / gg / gk",             "PhotoZ SWG",             "Tomographic bin files for tSZ cross-correlation"],
        ["Source n(z) for κκ / κy / gk",           "Weak Lensing SWG",       "Shear source bins n(z)"],
        ["f_sky — ACT × Euclid overlap",            "Maps SWG",               "Overlap mask or effective f_sky value"],
        ["Non-thermal pressure priors",             "tSZ / ICM SWG",          "α_nt, β_nt, n_nt constraints from X-ray / SZ data"],
        ["CLOE interface sign-off",                 "CLOE team",              "Data-vector format + covariance format validation"],
    ],
    col_widths=[4.2, 3.5, 5.4],
)

# ── Slide 9: Next steps ────────────────────────────────────────────────────────
sl = add_slide()
content_slide(sl, "Next steps before Consortium meeting", [
    "Merge Fix C — apply √S to C^gy and C^κy  (resolves Gaussian cov PSD)",
    "Validate C(ℓ) against CLASS-SZ or independent code for ≥ 2 probes",
    "Collect n(z) files from PhotoZ + WL SWGs",
    "Confirm f_sky with maps team",
    "Draft HOD priors — contact galaxy clustering SWG",
    "Run end-to-end NUTS chain (multi-probe) with regularised covariance",
    "CLOE interface test — confirm data-vector format",
    "Prepare 1-page summary of astrophysical parameters and their priors",
], note="Person-power needed: covariance validation, NaMaster integration, HOD calibration")

# ── Slide 10: Discussion ───────────────────────────────────────────────────────
sl = add_slide()
content_slide(sl, "Discussion points for Consortium meeting", [
    "HOD calibration: which simulations? Who owns this in the consortium?",
    "Non-thermal pressure: fixed profile or sampled over α_nt?",
    "Pseudo-Cℓ vs analytic covariance: is NaMaster needed for DR1?",
    "    If yes: who integrates NaMaster with GODMAX?",
    "Joint analysis with CLASS-SZ: shared data-vector? Shared covariance?",
    "Person-power: two bottlenecks identified",
    "    (1) Covariance validation + NaMaster integration",
    "    (2) HOD calibration vs Euclid simulations",
])

# ── save ──────────────────────────────────────────────────────────────────────
out = "/home/vtinnane/Documents/Codes/develop/GODMAX/notebooks/GODMAX_status_slides.pptx"
prs.save(out)
print("Saved:", out)
