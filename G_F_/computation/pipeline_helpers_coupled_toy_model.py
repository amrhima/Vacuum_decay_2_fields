#!/usr/bin/env python3
"""pipeline_helpers_coupled_toy_model.py -- shared I/O, background and plumbing of the
coupled_toy_model pipeline: bounce loading, the potential insertion, the 2x2
solution-matching determinant used to LOCATE the sector eigenvalues,
output-file naming, CLI, provenance stamps.

THE FIELD NUMBER IS FIXED AT TWO here (potential_coupled_toy_model.FIELDS =
('x','y')), so every matrix below is 2x2 and the bounce profile is (nR,2).  The
shapes are still written as (N,N)/(nR,N) in the docstrings because the code
carries N through unchanged -- N is simply always 2 in this pipeline.

Every helper here has >= 2 stage-file consumers and exactly ONE home (no
duplication anywhere).  Nothing in this module knows about a specific
stage; the physics lives in the stage files that import it.  The s2
integration scheme (the log-Simpson quadrature, the resolvent sector grid
and integral) lives in its OWN module, pipeline_quadrature_coupled_toy_model.

CONTENTS
--------
    bounce_arrays(npz)          raw (R, Phi', params, false_vac, fields)
                                of a bounce npz, with the LEGACY key fallback
                                (X_/Y_bounce_prime) -- the ONE schema reader
    load_bounce(bounce_npz)     bounce npz -> dict of the shared physics
                                background (R, Phi' (nR,2), potential,
                                Hessian masses m2 (2,), mu = sum_i m_i,
                                DW = H - diag(m2) (nR,2,2),
                                trDW3 = tr(DW^3), fields, N)
    read_band_tables(...)       glob + merge the residual-band slice npz set
                                (base slices and _n<lo>to<hi> wave-extension
                                files) into {s2: {n: delta_geq3}}; fail-loud
                                on provenance mismatch or conflicting
                                duplicates (band_adaptive + tail_s2_completion)
    potential_insertion_V(...)  potential insertion V(r) = H(phi_b) - H_FV
                                (full 2x2 + diagonal), for the counterterms
    sector_matching_det(band, n, s2)
                                2x2 solution-matching determinant det(1+M)
                                at rmax; a bound state sits where it crosses
                                zero -- used to LOCATE the sector poles
                                (an eigenvalue condition, not a
                                determinant value)
    require_finite(arr, msg)    fail-loud NaN/inf check (nansum is banned)
    atomic_savez(path, **kwds)  crash-safe np.savez (temp file + os.replace)
    stage_paths(data_dir, tag)  the canonical output npz path of every stage
    add_standard_cli(parser)    the uniform --data-dir --bounce-npz --tag
    bounce_sha256(path)         sha256 of the bounce npz bytes (provenance)
    potential_id(m2)            short id of the potential in use (provenance;
                                separates the two hardcoded models and
                                catches an edited coupling)
    provenance_stamp(...)       the three metadata keys every stage stamps

INPUTS/OUTPUTS: no hidden state; file access is confined to load_bounce /
bounce_sha256 (the bounce npz), read_band_tables (band-slice npz + the
manifest json), potential_id (the potential module source) and atomic_savez
(writing stage outputs).  `import potential_coupled_toy_model` happens INSIDE
load_bounce/potential_id (lazy), so importing this module never needs the
potential module.
"""
import hashlib
import os
import numpy as np

# provenance/robustness metadata (not physics): stamped as `code_version`
# into every stage output npz (the band engine stamps its own
# COUPLED_TOY_MODEL_ENGINE_VERSION -- see delta_g_bar_greater_equal_3_coupled_toy_model).
COUPLED_TOY_MODEL_PIPELINE_VERSION = 'coupled_toy_model.1'


# --------------------------------------------------------------------------- #
#  Bounce / potential background (the shared physics input of every stage)     #
# --------------------------------------------------------------------------- #
def bounce_arrays(b):
    """The raw arrays of an OPEN bounce npz `b`, under the coupled_toy_model
    schema with the LEGACY key fallback (the ONE schema reader):
        R (nR,), Phi' (nR,2), params (15,), false_vac (2,), fields ('x','y').
    New schema: `Phi_bounce_prime` (nR,2) + `fields`.  Legacy fallback: stack
    X_bounce_prime/Y_bounce_prime; a missing `fields` key is simply the fixed
    pair ('x','y') -- this pipeline has no other.  The shapes are validated
    against the two fields and the 15 degree-<=4 monomials (fail-loud), so a
    bounce npz from a different (e.g. three-field) tree is rejected rather
    than half-read."""
    import potential_coupled_toy_model as potential_mod
    R = np.asarray(b['R'], float)
    false_vac = np.asarray(b['false_vac'], float)
    if 'Phi_bounce_prime' in b.files:
        Phi = np.asarray(b['Phi_bounce_prime'], float)
    elif 'X_bounce_prime' in b.files and 'Y_bounce_prime' in b.files:
        Phi = np.stack([np.asarray(b['X_bounce_prime'], float),
                        np.asarray(b['Y_bounce_prime'], float)], axis=1)
    else:
        raise RuntimeError('[ABORT] bounce npz has neither Phi_bounce_prime '
                           'nor the legacy X_/Y_bounce_prime keys.')
    fields = (tuple(str(f) for f in np.asarray(b['fields']).ravel())
              if 'fields' in b.files else potential_mod.FIELDS)
    if fields != potential_mod.FIELDS:
        raise RuntimeError(
            f'[ABORT] bounce npz was solved for fields {fields}, but this '
            f'pipeline has exactly the two fields {potential_mod.FIELDS}.')
    params = np.asarray(b['params'], float)
    if params.size != potential_mod.N_PARAMS:
        raise RuntimeError(
            f'[ABORT] bounce npz params has {params.size} entries but the two '
            f'fields {fields} need {potential_mod.N_PARAMS} (the degree-<=4 '
            f'monomial coefficients) -- the npz was solved for a different '
            f'field set.')
    if Phi.ndim != 2 or Phi.shape[1] != len(fields):
        raise RuntimeError(f'[ABORT] bounce profile shape {Phi.shape} does '
                           f'not match fields {fields}.')
    return R, Phi, params, false_vac, fields


def load_bounce(bounce_npz):
    """Load the bounce npz and build the shared background every stage uses.

    Returns a dict (N = 2, the two coupled scalar fields):
        R       (nR,)      radial grid
        Phi     (nR,N)     primed-frame bounce profile phi'_b(r)
        fields  (N,)       ('x','y')
        N       ()         2
        pot                CTShiftedLiftedPotential (the stored potential)
        Hfv     (N,N)      false-vacuum Hessian (diagonal; masses^2)
        m2      (N,)       channel masses^2 (diagonal of Hfv)
        mbar2   ()         min(m2) (lightest channel; completion/fict scale)
        masses  list(m2)   python list (counterterm convention)
        mu      ()         renormalization scale mu = sum_i m_i
        DW      (nR,N,N)   Hessian insertion H(phi_b(r)) - diag(m2)
        trDW3   (nR,)      tr(DW^3) (analytic-A3 / Baacke-c / completion)
    """
    import potential_coupled_toy_model as potential_mod
    if not bounce_npz:
        raise RuntimeError('[ABORT] --bounce-npz (or COUPLED_TOY_MODEL_BOUNCE_NPZ) is required.')
    b = np.load(bounce_npz, allow_pickle=True)
    R, Phi, params, false_vac, fields = bounce_arrays(b)
    N = len(fields)
    pot = potential_mod.CTShiftedLiftedPotential(params, false_vac)
    # ---- BOUNCE-BASIS guard (does NOT enter lnD) -----------------------------
    # The stored PRIMED profile lives in the false-vacuum Hessian eigenbasis,
    # whose eigenvectors (sign, ordering, rotation inside a degenerate
    # eigenspace) are NOT unique -- `pot` rebuilds that basis here with eigh, and
    # on a different LAPACK/platform or a degenerate-mass H_F it may differ from
    # the one that generated the stored prime, making the trusted prime
    # inconsistent with `pot`.  The ORIGINAL-frame profile is canonical
    # (basis-free), so reconstruct the prime from it through THIS pot's basis and
    # fail loud if the stored prime disagrees.  (For reproducible bounces the two
    # coincide to ~1e-15, so the stored Phi is kept and results stay identical.)
    Phi_orig = None
    if 'Phi_bounce_orig' in b.files:
        Phi_orig = np.asarray(b['Phi_bounce_orig'], float)
    elif 'X_bounce_orig' in b.files and 'Y_bounce_orig' in b.files:
        Phi_orig = np.stack([np.asarray(b['X_bounce_orig'], float),
                             np.asarray(b['Y_bounce_orig'], float)], axis=1)
    if Phi_orig is not None and Phi_orig.shape == Phi.shape:
        dev = float(np.max(np.abs(Phi - pot.to_prime(Phi_orig))))
        if dev > 1e-8:
            raise RuntimeError(
                f'[ABORT BOUNCE-BASIS] the stored primed profile disagrees with '
                f'the one reconstructed from the canonical original-frame profile '
                f'through the recomputed false-vacuum mass basis (max dev '
                f'{dev:.2e} > 1e-8) -- the eigenbasis is not reproducible here '
                f'(degenerate masses or a different LAPACK/platform).  Regenerate '
                f'the bounce so the primed profile matches the local basis.')
    Hfv = pot.H(np.zeros(N))
    m2 = np.array([float(Hfv[i, i]) for i in range(N)])
    mbar2 = float(m2.min())
    masses = [float(v) for v in m2]
    mu = float(np.sum(np.sqrt(m2)))
    DW = np.array([pot.H(Phi[k]) - np.diag(m2) for k in range(len(R))])
    trDW3 = np.einsum('kij,kjl,kli->k', DW, DW, DW)
    return dict(R=R, Phi=Phi, fields=fields, N=N, pot=pot, Hfv=Hfv, m2=m2,
                mbar2=mbar2, masses=masses, mu=mu, DW=DW, trDW3=trDW3)


def wall_radius(R, Phi):
    """r_W = R at max |dPhi'/dr| (vector norm) -- the ONE owner of the
    wall-radius rule, used by watchers_guards.analytic_moments and by the
    band engine's --n-max fallback.  Accepts (nR,) or (nR, N)."""
    R = np.asarray(R, float)
    Phi = np.asarray(Phi, float)
    if Phi.ndim == 1:
        Phi = Phi[:, None]
    dPhi = np.gradient(Phi, R, axis=0)
    speed = np.sqrt(np.sum(dPhi * dPhi, axis=1))
    return float(R[np.argmax(speed)])


def load_eig_npz(path_or_none, data_dir, tag, key, bounce_npz, rerun_hint):
    """Read + validate a stage-2 eigenvalue npz (the ONE loader shared by the
    two sector stages): resolve the default path, abort if missing, and abort
    if it was computed against a DIFFERENT bounce (bounce_sha mismatch)."""
    eig_path = path_or_none or stage_paths(data_dir, tag)[key]
    if not os.path.isfile(eig_path):
        raise RuntimeError(f'[ABORT] eigenvalue npz missing: {eig_path} -- '
                           f'run {rerun_hint} first.')
    eig = np.load(eig_path, allow_pickle=True)
    cur_sha = bounce_sha256(bounce_npz)
    eig_sha = str(eig['bounce_sha']) if 'bounce_sha' in eig.files else ''
    if cur_sha and eig_sha and eig_sha != cur_sha:
        raise RuntimeError(
            f'[ABORT] {key} npz was computed against a DIFFERENT bounce '
            f'(bounce_sha {eig_sha[:8]} != {cur_sha[:8]}) -- rerun '
            f'{rerun_hint} or point --bounce-npz at the right file.')
    return eig


def read_band_tables(data_dir, tag, expect_sha=None, expect_tol=None,
                     manifest_path=None, s2_cap=None):
    """Glob and merge every residual-band slice npz of this tag -- the base
    slices residual_band_<tag>_s2<v>.npz AND the wave-extension files
    residual_band_<tag>_s2<v>_n<lo>to<hi>.npz the adaptive driver writes --
    into one lookup  tables[s2][n] = delta_geq3.

    Fail-loud rules (no silent mixing):
      - expect_sha: every file's bounce_sha must equal it (a same-tag slice
        from another bounce aborts; a file without the stamp aborts too);
      - expect_tol=(rtol, atol): the engine tolerances must match;
      - declared wave thinning (n_stride > 1) aborts -- the adaptive driver
        generates full bands only;
      - a wave present twice for the same s2 must agree to 1e-10 relative
        (else one of the files is stale -- abort);
      - s2_cap (if given): slice files ABOVE the cap are excluded up front --
        they are beyond-rectangle data (e.g. a resumed run whose watcher
        stopped below existing data) and never enter this read;
      - manifest_path (if the file exists): the kept set must equal the
        manifest set exactly (the MANIFEST guard -- a foreign same-tag slice
        would otherwise be summed in silently).
    Returns (s2_values sorted ascending, tables, file_list)."""
    import glob
    files = sorted(glob.glob(os.path.join(
        data_dir, f'residual_band_{tag}_s2*.npz')))
    if s2_cap is not None:
        kept = []
        for f in files:
            with np.load(f, allow_pickle=True) as d:
                if float(d['s2']) <= s2_cap + 1e-9:
                    kept.append(f)
        files = kept
    if manifest_path and os.path.isfile(manifest_path):
        import json
        with open(manifest_path) as fh:
            man = json.load(fh)
        want = set(os.path.basename(x) for x in
                   (man.get('slices', man) if isinstance(man, dict) else man))
        have = set(os.path.basename(f) for f in files)
        extra, missing = sorted(have - want), sorted(want - have)
        if extra or missing:
            raise RuntimeError(
                f'[ABORT MANIFEST] band slice set does not match '
                f'{os.path.basename(manifest_path)}: EXTRA (foreign) {extra}; '
                f'MISSING {missing}.  Remove the foreign slice(s) or '
                f'regenerate the band.')
    tables = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        if expect_sha is not None:
            sha = str(d['bounce_sha']) if 'bounce_sha' in d.files else ''
            if sha != expect_sha:
                raise RuntimeError(
                    f'[ABORT BAND-TABLES] {os.path.basename(f)} was computed '
                    f'against a different bounce (sha {sha[:12]}... vs '
                    f'{expect_sha[:12]}...) -- delete or retag it.')
        if expect_tol is not None:
            if 'rtol' not in d.files:
                raise RuntimeError(
                    f'[ABORT BAND-TABLES] {os.path.basename(f)} carries no '
                    f'engine-tolerance stamp (pre-provenance file) -- delete '
                    f'or retag it.')
            if (float(d['rtol']), float(d['atol'])) != tuple(expect_tol):
                raise RuntimeError(
                    f'[ABORT BAND-TABLES] {os.path.basename(f)} used engine '
                    f'tolerances ({float(d["rtol"]):g}, {float(d["atol"]):g}) '
                    f'!= expected {expect_tol} -- delete or retag it.')
        if 'n_stride' in d.files and int(d['n_stride']) > 1:
            raise RuntimeError(
                f'[ABORT BAND-TABLES] {os.path.basename(f)} declares wave '
                f'thinning (n_stride={int(d["n_stride"])}); the adaptive '
                f'driver works on full bands only -- regenerate.')
        s2 = float(d['s2'])
        row = tables.setdefault(s2, {})
        for n, v in zip(np.asarray(d['n_values'], int),
                        np.asarray(d['delta_geq3'], float)):
            n = int(n)
            if n in row and abs(v - row[n]) > 1e-10 * max(abs(v), 1e-30):
                raise RuntimeError(
                    f'[ABORT BAND-TABLES] wave n={n} at s2={s2:g} appears in '
                    f'two slice files with conflicting values ({row[n]!r} vs '
                    f'{v!r}) -- a stale extension file; delete it.')
            row[n] = float(v)
    return np.array(sorted(tables.keys())), tables, files


def potential_insertion_V(R, Phi, pot, Hfv):
    """Potential insertion V(r) = H(phi_b(r)) - H_FV on the bounce grid.
    Phi is the (nR,N) primed-frame profile.  Returns (V_full (nR,N,N),
    V_diag (N,nR)) -- the counterterm integrands (A1 uses the diagonal,
    A2 the full 2x2)."""
    N = np.shape(Hfv)[0]
    V_full = np.zeros((len(R), N, N))
    for k in range(len(R)):
        V_full[k] = pot.H(np.asarray(Phi[k], float)) - Hfv
    V_diag = np.array([V_full[:, i, i] for i in range(N)])
    return V_full, V_diag


# --------------------------------------------------------------------------- #
#  Sector eigenvalue locator (2x2 solution-matching determinant)               #
# --------------------------------------------------------------------------- #
def sector_matching_matrix(band, n, s2):
    """The 2x2 solution-matching matrix I + b1 + b2 + b3 whose determinant is
    the bound-state condition (sector_matching_det).  Exposed so a located
    crossing can be tested for MULTIPLICITY -- the number of modes that are
    EXACTLY degenerate there = dim of the matrix null space -- which the scalar
    determinant alone cannot resolve.  `band` is a ResidualBand instance
    (duck-typed; no band-engine import)."""
    y = band.h_inf(n, float(s2), '-')
    b1_, b2_, b3_ = y[2].T, y[4].T, y[6].T
    return np.eye(b1_.shape[0]) + b1_ + b2_ + b3_


def sector_matching_det(band, n, s2):
    """2x2 solution-matching determinant; a bound state sits where it
    crosses zero; used to LOCATE sector poles -- an eigenvalue condition,
    not a determinant value.  `band` is a ResidualBand instance
    (duck-typed; no band-engine import); the channel count is read off the
    returned boundary block."""
    return float(np.linalg.det(sector_matching_matrix(band, n, s2)))


def crossing_multiplicity(band, n, s2, tol=1e-4):
    """Number of modes that are EXACTLY degenerate at a matching-det crossing =
    dim of the null space of the matching matrix M = I + b1 + b2 + b3 there, read
    from its singular values.  A genuine index-contributing crossing is SIMPLE
    (multiplicity 1); multiplicity >= 2 means several modes coincide at this s2 --
    a symmetric multi-wall composite whose degenerate modes the scalar det's SIGN
    change cannot count (an ODD multiplicity shows a single sign change and would
    be falsely certified index-1).

    The reference scale is max(1, sv_max): M carries an identity, so a
    non-vanishing direction has sv ~ O(1) while a mode's vanishing direction has
    sv -> 0.  Referencing to 1 (not sv_max) is essential for a FULLY degenerate
    crossing, where every sv vanishes and sv_max itself is ~0.  Vanishing sv
    (~1e-6 or below at a brentq-located root) sit far under tol; non-vanishing
    O(1) sv sit far above -- a wide, non-fragile margin."""
    sv = np.linalg.svd(sector_matching_matrix(band, n, float(s2)),
                       compute_uv=False)
    if sv.size == 0:
        return 0
    ref = max(1.0, float(sv[0]))
    return int(np.sum(sv < tol * ref))


def _scan_det_svmin(band, n, scan):
    """det(M) AND the normalized smallest singular value of M = I+b1+b2+b3 at
    every scan point -- both from the SAME h_inf solve per point (no extra ODE
    cost).  svmin_norm = sv_min / max(1, sv_max) dips toward 0 near EVERY root
    of the matching problem, including roots the det SIGN cannot see (an even
    number of roots inside one interval flips the sign twice = not at all)."""
    dets = np.empty(len(scan))
    svmn = np.empty(len(scan))
    for k, s2 in enumerate(scan):
        M = sector_matching_matrix(band, n, s2)
        dets[k] = float(np.linalg.det(M))
        sv = np.linalg.svd(M, compute_uv=False)
        svmn[k] = float(sv[-1]) / max(1.0, float(sv[0]))
    return dets, svmn


def stable_crossing_scan(band, n, make_scan, n0=33, cap=1025):
    """Self-certifying root census of the sector matching det, in two layers.

    make_scan(npts) -> the s2 scan array at a given density (the CALLER owns
    the window shape: geometric for n=0, the dense-near-zero window for n=1).

    LAYER 1 (sign count at certified density): a FIXED-density scan can hide
    two nearby roots inside one interval (their two sign flips cancel), so
    the density is DOUBLED until two consecutive densities count the SAME
    number of sign changes.  A det that is EXACTLY zero on a scan point is
    the root itself and counts as one crossing.

    LAYER 2 (hidden-root sweep): the det SIGN is blind to an EVEN number of
    roots in one interval at ANY density, but every root makes M lose rank,
    so the normalized smallest singular value svmin (computed per point from
    the same solves) dips locally near every root.  Each pronounced svmin
    dip NOT adjacent to a counted sign change is refined with a dense LINEAR
    sub-scan of its neighborhood: sign changes found there are hidden roots
    (a near-split pair), and a dip that reaches rank loss (svmin < 1e-3)
    without any sign change is a tangential EVEN-multiplicity root pair.
    Both add to the census (over-rejecting is safe; falsely certifying
    index-1 is not).

    Returns (scan, dets, flips, n_hidden): flips = indices i with a sign
    change in (scan[i], scan[i+1]) on the certified grid; n_hidden = roots
    found only by layer 2.  Aborts (fail-loud) if the sign count has not
    stabilized by `cap` points."""
    prev = None
    npts = int(n0)
    while True:
        scan = np.asarray(make_scan(npts), float)
        dets, svmn = _scan_det_svmin(band, n, scan)
        require_finite(dets, f'[ABORT COLEMAN-SCAN] non-finite det(1+M) on '
                             f'the n={n} census scan')
        sgn = np.sign(dets)
        exact = int(np.sum(dets == 0.0))
        flips = np.where(sgn[:-1] * sgn[1:] < 0.0)[0]
        count = int(flips.size) + exact
        if prev is not None and count == prev:
            break
        if npts >= cap:
            raise RuntimeError(
                f'[ABORT COLEMAN-SCAN] the n={n} det(1+M) crossing count has '
                f'not stabilized by {npts} scan points (last counts '
                f'{prev} -> {count}) -- the spectrum has roots closer than '
                f'the scan can resolve; inspect this bounce/potential.')
        prev = count
        npts = 2 * npts - 1          # doubled density, same endpoints

    # ---- layer 2: refine every suspicious svmin dip --------------------------
    flip_adjacent = set(int(i) for i in flips) | set(int(i) + 1 for i in flips)
    med = float(np.median(svmn))
    n_hidden = 0
    for i in range(1, len(scan) - 1):
        is_dip = (svmn[i] < svmn[i - 1] and svmn[i] <= svmn[i + 1]
                  and svmn[i] < 0.2 * med)
        if not is_dip or i in flip_adjacent:
            continue
        sub = np.linspace(scan[i - 1], scan[i + 1], 129)
        d2, s2m = _scan_det_svmin(band, n, sub)
        sg2 = np.sign(d2)
        found = int(np.where(sg2[:-1] * sg2[1:] < 0.0)[0].size)
        if found == 0 and float(np.min(s2m)) < 1e-3:
            found = 2          # tangential rank loss with no sign change
        if found:
            print(f'  [COLEMAN-SCAN n={n}] svmin dip at s2~{scan[i]:.4g} '
                  f'refined: {found} hidden root(s) the det sign missed')
        n_hidden += found
    return scan, dets, flips, n_hidden


def require_finite(arr, msg):
    """Fail-loud non-finite check: np.sum-style, NEVER nansum (a silent
    NaN -> 0 hides a failed partial wave)."""
    arr = np.asarray(arr)
    bad = ~np.isfinite(arr)
    if bad.any():
        raise RuntimeError(f'{msg} ({int(bad.sum())}/{arr.size} non-finite '
                           f'values -- fail-loud, no nansum).')


def atomic_savez(path, **kwds):
    """Save an .npz the crash-safe way: write to a TEMP file in the SAME
    directory first, then rename it onto the final name with os.replace.
    os.replace is an atomic rename on one filesystem, so a crash (or a full
    disk) mid-write can never leave a HALF-WRITTEN .npz where the pipeline
    later expects a complete one -- either the old file stays untouched or
    the new one appears whole.  The temp name is DOT-PREFIXED so it can never
    match the residual_band glob (a killed worker must not leave a
    half-written file that reads as a slice).  The keys/values written are
    exactly what np.savez(path, **kwds) would write."""
    final = path if str(path).endswith('.npz') else f'{path}.npz'
    tmp = os.path.join(os.path.dirname(final) or '.',
                       f'.{os.path.basename(final)}.tmp{os.getpid()}.npz')
    np.savez(tmp, **kwds)
    os.replace(tmp, final)


# --------------------------------------------------------------------------- #
#  Output naming / CLI / provenance (metadata, never physics)                  #
# --------------------------------------------------------------------------- #
def stage_paths(data_dir, tag):
    """The canonical output npz path of every pipeline stage (ONE naming
    convention; the assembler and the orchestrator skip-guards use these).

    Every name is <stage>_<tag>.npz and NOTHING is appended to the tag: the tag
    itself is required to contain 'coupled_toy_model' (the driver aborts if it
    does not), so every file this pipeline writes already announces which
    pipeline produced it -- exactly once.  The band engine's slice files
    (residual_band_<tag>_s2<v>.npz) and the band manifest follow the same rule.
    """
    return {
        'bounce':         os.path.join(data_dir, f'bounce_data_{tag}.npz'),
        'band_cutoffs':   os.path.join(data_dir, f'band_cutoffs_{tag}.npz'),
        'eig_n0':         os.path.join(data_dir, f'eig_n0_{tag}.npz'),
        'eig_n1':         os.path.join(data_dir, f'eig_n1_{tag}.npz'),
        'sector_n0':      os.path.join(data_dir, f'sector_n0_{tag}.npz'),
        'sector_n1':      os.path.join(data_dir, f'sector_n1_{tag}.npz'),
        'band_integrals': os.path.join(data_dir, f'band_integrals_{tag}.npz'),
        'tail_highn':     os.path.join(data_dir, f'tail_highn_{tag}.npz'),
        'ct_tadpole':     os.path.join(data_dir, f'ct_tadpole_{tag}.npz'),
        'ct_fish':        os.path.join(data_dir, f'ct_fish_{tag}.npz'),
        'D_integral':     os.path.join(data_dir, f'D_integral_{tag}.npz'),
    }


def add_standard_cli(parser):
    """The uniform CLI prefix of every stage file:
    --data-dir --bounce-npz --tag (stage-specific flags follow)."""
    parser.add_argument('--data-dir',
                        default=os.environ.get('G_PROJECT_DATA', '.'),
                        help='data directory (all stage npz live here; '
                             'default $G_PROJECT_DATA)')
    parser.add_argument('--bounce-npz',
                        default=os.environ.get('COUPLED_TOY_MODEL_BOUNCE_NPZ'),
                        help='bounce npz path (R, Phi\', params, false_vac, '
                             'fields; default $COUPLED_TOY_MODEL_BOUNCE_NPZ)')
    parser.add_argument('--tag', default='coupled_toy_model',
                        help='run tag, threaded into every filename '
                             '(must contain "coupled_toy_model")')
    return parser


def bounce_sha256(path):
    """sha256 of the bounce npz BYTES (the exact file this run integrated
    against).  provenance/robustness metadata (not physics)."""
    if not path or not os.path.isfile(path):
        return ''
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def potential_id(m2):
    """Short string identifying the potential this run integrated against:
    round(m^2,6) of BOTH channels + a sha256 prefix of the potential module
    SOURCE.  The channel masses alone already separate the two hardcoded
    models (coupled F2_T0: m^2 = 2.8879, 7.0917; decoupled bdet: 6.7551,
    6.7551), and the source hash additionally catches an edited coupling.
    provenance/robustness metadata (not physics)."""
    try:
        import potential_coupled_toy_model as potential_mod
        src = ''
        pf = getattr(potential_mod, '__file__', None)
        if pf and os.path.isfile(pf):
            with open(pf, 'rb') as fh:
                src = hashlib.sha256(fh.read()).hexdigest()[:12]
    except Exception:
        src = ''
    m2a = np.asarray(m2, float).ravel()
    m2s = ','.join(str(round(float(v), 6)) for v in m2a)
    return f"m2={m2s}|src={src}"


def provenance_stamp(bounce_npz, m2):
    """The three metadata keys every stage output npz carries."""
    return dict(bounce_sha=bounce_sha256(bounce_npz),
                potential_id=potential_id(m2),
                code_version=COUPLED_TOY_MODEL_PIPELINE_VERSION)
