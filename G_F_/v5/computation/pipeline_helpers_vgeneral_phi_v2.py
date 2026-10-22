#!/usr/bin/env python3
"""pipeline_helpers_vgeneral_phi_v2.py -- the library every stage code, the
add-on and the tests import (no level of its own in the README tree): the
bounce BACKGROUND every level computes on (the interaction Hessian U_int),
the read rules of the stage files, file naming, the common CLI, the
provenance stamps and the index bookkeeping of the Green's-function
assembly.  No physics STAGE lives here; the one convention set here is the
MSbar scale mu = sum_i m_i (channel_masses_and_mu).  The field number N is
never hardcoded: it comes from the bounce npz (`fields`) and is threaded
through every array shape.

READS
    bounce_data_<tag>.npz (stage 0, bounce_vgeneral_phi_v2): R,
        phi_b (the profile in the primed frame), params, fields, false_vac,
        true_vac, mass_basis_L, pair_name
    every stage npz (load_stage_npz): its bounce_sha stamp
    residual_band_<tag>_s2<v>[_n<lo>to<hi>].npz, the band engine's slices
        (read_band_tables): s2, n_values, delta_geq3 and the stamps
        bounce_sha, potential_id, m2, code_version, rtol, atol;
        band_manifest_<tag>.json (the slice set the band driver wrote)
WRITES
    nothing of its own: atomic_savez is the ONE write rule of every stage
    npz, provenance_stamp the three metadata keys (bounce_sha, potential_id,
    code_version) every stage npz after the bounce carries.

HOW -- each name has exactly ONE home, this module:
    read_bounce_npz(bounce_data)  the schema reader of an open bounce npz ->
                                (R, phi_b, params, false_vac, fields)
    load_background(bounce_npz) -> dict {R, phi_b (nR,N, primed frame),
                                fields, N, params, pot_prime, m2, m_min2,
                                mu = sum_i m_i, U_int = H'(phi_b) - diag(m2),
                                tr_U_int3 = tr(U_int^3)}: THE background
                                every stage computes on; guards BOUNCE-BASIS
                                and FV-STATIONARITY
    FALSE_VAC_OFFSET_MAX / fv_newton_step(pot_prime)
                                THE stationarity rule of the false vacuum:
                                one Newton step |H'(0)^-1 dV'(0)| against
                                the vacuum separation
    channel_masses_and_mu(pot_prime)   m2 = PrimedPotential.m2 and the MSbar
                                scale mu = sum_i m_i; guard FV-MASS-POSITIVE
    insertion(pot_prime, phi_b)   the interaction Hessian (the insertion)
                                U_int(r) = H'(phi_b(r)) - diag(m2), (nR, N, N)
    coupled_pairs(U_int)        the channel pairs the insertion couples,
                                (N, N) bool (ONE round-off rule: the solve
                                radius and the h-ODE engine)
    trace_U_int_cubed(U_int)    tr U_int^3(r) on the grid; its radial moments
                                set the high-s2 tail, A3 and baacke_c
    wall_radius(R, phi_b)       r_W = R at max ||dphi_b/dr|| (ONE wall rule)
    load_stage_npz(...)         read + fingerprint-check ANY stage npz (the
                                one read rule of the stage files)
    read_band_tables(...)       glob + merge the residual-band slices into
                                {s2: {n: delta_geq3}}; fail-loud provenance
    require_finite / atomic_savez / stage_paths / add_standard_cli /
    bounce_fingerprint / potential_id / provenance_stamp    (I/O + provenance)
    VGENERAL_PHI_PIPELINE_VERSION / VGENERAL_PHI_ENGINE_VERSION   the build
                                stamps (code_version) of every stage npz and
                                of every band slice (read_band_tables checks
                                the latter)
    s2_filetag(s2) / bounce_name(npz)   the two filename pieces every figure
                                and s2-parametrised npz (except the band
                                slices) carries
    band_s2_text(s2)            the text of one band s2 node (the ladder, the
                                engine command line, the slice file name)
    radius_first(y)             solver block (..., n_r) -> stack (n_r, ...)
                                (array layout only)
    bracket_outer               the coincident bracket of the assembled
                                Green's function, G_bracket = (1+h^+) C^-T
                                (1+h^-)^T, as an einsum index sum
    bracket_outer_pairs / bracket_inner_pairs   the two index placements on
                                two radial grids (the off-coincident add-on)

The mode census (every s2 where an eigenvalue of the constant Wronskian C
vanishes) lives in mode_census_vgeneral_phi_v2, the s2 quadrature in
pipeline_quadrature_vgeneral_phi_v2.  The potential module is imported lazily
(only calls that build the model need it).
"""
import hashlib
import os
import re
import numpy as np

# The two build stamps (provenance, not physics), both written as `code_version`:
# VGENERAL_PHI_PIPELINE_VERSION -- provenance_stamp (and bounce_vgeneral_phi_v2
# directly) writes it into every STAGE npz; no stage checks it.
# VGENERAL_PHI_ENGINE_VERSION -- the BAND ENGINE writes it into every residual_band
# slice and read_band_tables below refuses a slice with another stamp.  Bump the
# engine stamp whenever what a slice holds changes -- its NUMBERS by more than
# the engine's integration tolerance (LSODA_RTOL, LSODA_ATOL of
# greens_function), its keys or its s2 nodes -- then regenerate the stored
# bands.
VGENERAL_PHI_PIPELINE_VERSION = 'vgeneral_phi_v2.4'
VGENERAL_PHI_ENGINE_VERSION = 'vgeneral_phi_v2.4'


# --------------------------------------------------------------------------- #
#  Bounce / potential background (the shared physics input of every stage)     #
# --------------------------------------------------------------------------- #
def read_bounce_npz(bounce_data):
    """The raw arrays of an OPEN bounce npz `bounce_data`:
        R (nR,), phi_b (nR,N) the primed profile,
        params, false_vac (N,), fields (tuple of str).
    A bounce npz without the keys `phi_b` and `fields` is
    rejected; the params length is checked against the degree-<=4 monomial
    count C(N+4,4) of those fields, and the profile shape against N."""
    import potential_vgeneral_phi_v2 as potential_mod
    R = np.asarray(bounce_data['R'], float)
    false_vac = np.asarray(bounce_data['false_vac'], float)
    if 'phi_b' not in bounce_data.files or 'fields' not in bounce_data.files:
        raise RuntimeError(
            '[ABORT BOUNCE-SCHEMA] bounce npz lacks phi_b and/or '
            'fields -- it is not a current-schema bounce.  Re-solve it with '
            'bounce_vgeneral_phi_v2.')
    phi_b = np.asarray(bounce_data['phi_b'], float)
    fields = tuple(str(f) for f in np.asarray(bounce_data['fields']).ravel())
    params = np.asarray(bounce_data['params'], float)
    want = potential_mod.n_params(fields)
    if params.size != want:
        raise RuntimeError(
            f'[ABORT] bounce npz params has {params.size} entries but fields '
            f'{fields} need {want} (C(N+4,4) degree-<=4 monomials) -- the npz '
            f'was solved for a different field set.')
    if phi_b.ndim != 2 or phi_b.shape[1] != len(fields):
        raise RuntimeError(f'[ABORT] bounce profile shape {phi_b.shape} does '
                           f'not match fields {fields}.')
    return R, phi_b, params, false_vac, fields


def load_background(bounce_npz):
    """Bounce npz -> the shared background dict every stage uses:
        R        (nR,)     radial grid
        phi_b    (nR,N)    bounce profile phi'_b(r), PRIMED frame
        fields, N, params  the model the bounce was solved with
        pot_prime          PrimedPotential rebuilt from (params, false_vac)
        m2, m_min2         channel masses^2 = pot_prime.m2, and min(m2)
        mu       ()        MSbar scale mu = sum_i m_i
        U_int      (nR,N,N)  the interaction Hessian (insertion)
                           U_int = H'(phi_b) - diag(m2)
        tr_U_int3  (nR,)     tr(U_int^3)(r): the moment behind A3 and
                           baacke_c (watchers_guards.analytic_moments) and
                           the high-s2 tail shape
                           (pipeline_quadrature.high_s2_tail)
    """
    import potential_vgeneral_phi_v2 as potential_mod
    if not bounce_npz:
        raise RuntimeError('[ABORT] --bounce-npz (or VGENERAL_PHI_BOUNCE_NPZ) is required.')
    bounce_data = np.load(bounce_npz, allow_pickle=True)
    R, phi_b, params, false_vac, fields = read_bounce_npz(bounce_data)
    true_vac = np.asarray(bounce_data['true_vac'], float)
    N = len(fields)
    pot_prime = potential_mod.PrimedPotential(params, false_vac, fields)
    # ---- BOUNCE-BASIS guard: the primed profile lives in the FV-Hessian
    # eigenbasis, which eigh does not fix uniquely (sign/degenerate rotation):
    # the solve-time basis mass_basis_L stamped by bounce_vgeneral_phi_v2
    # must equal the basis recomputed here.
    L_stored = np.asarray(bounce_data['mass_basis_L'], float)
    dev_L = float(np.max(np.abs(L_stored - pot_prime.L)))
    if dev_L > 1e-10:
        raise RuntimeError(
            f'[ABORT BOUNCE-BASIS] the mass basis recomputed here '
            f'differs from the solve-time mass_basis_L stamped in the '
            f'bounce npz (max dev {dev_L:.2e} > 1e-10) -- eigenbasis '
            f'not reproducible here (degenerate masses or different '
            f'LAPACK).  Regenerate the bounce.')
    # ---- FV-STATIONARITY guard: everything assumes grad V(phi_F) = 0 (phi_F
    # is the point the bounce's tail decays to AND the subtraction point of
    # U_int).  ONE rule, the same as bounce_vgeneral_phi_v2's FALSE-VACUUM
    # guard: the Newton step fv_newton_step against FALSE_VAC_OFFSET_MAX x
    # the vacuum separation (scale-free).
    vacuum_separation = float(np.linalg.norm(true_vac - false_vac))
    fv_newton_offset = fv_newton_step(pot_prime)
    if fv_newton_offset > FALSE_VAC_OFFSET_MAX * vacuum_separation:
        raise RuntimeError(
            f'[ABORT FV-STATIONARITY] one Newton step moves the stored false '
            f'vacuum {np.array2string(false_vac, precision=10)} by {fv_newton_offset:.3e} '
            f'= {fv_newton_offset / vacuum_separation:.2e} of the vacuum separation {vacuum_separation:.6g} (limit '
            f'{FALSE_VAC_OFFSET_MAX:g}) -- phi_F is not a stationary point of '
            f'V, so the point the bounce decays to and the U_int = H(phi_b) - '
            f'H(phi_F) subtraction point are both wrong.  Re-solve the vacuum '
            f'for this potential (bounce_vgeneral_phi_v2).')

    # the FV channel masses^2: PrimedPotential rotated H(phi_F) into its
    # eigenbasis (numpy.linalg.eigh), so H'(0) = diag(m2) and the free problem
    # separates into N Bessel channels of mass m_i
    m2, mu = channel_masses_and_mu(pot_prime)
    # m_min2 = the lightest mass^2 of ALL channels, idle ones included (the
    # m_min of bounce_polish fv_mass_min runs over the channels the bounce
    # acts on only, so m_min^2 > m_min2 when the lightest channel is idle)
    m_min2 = float(m2.min())
    U_int = insertion(pot_prime, phi_b)
    tr_U_int3 = trace_U_int_cubed(U_int)
    return dict(R=R, phi_b=phi_b, fields=fields, N=N, params=params,
                pot_prime=pot_prime, m2=m2,
                m_min2=m_min2, mu=mu, U_int=U_int, tr_U_int3=tr_U_int3)


# Max false-vacuum displacement from a true stationary point, as a fraction of
# the vacuum separation (the FALSE-VACUUM guard of bounce_vgeneral_phi_v2, its
# vacuum matching, and the FV-STATIONARITY guard of load_background).
FALSE_VAC_OFFSET_MAX = 1e-6


def fv_newton_step(pot_prime):
    """|delta phi_F| of ONE Newton step for grad V = 0 at the primed origin,
    delta = H'(0)^-1 dV'(0) (numpy.linalg.solve; the norm is the same in both
    frames, L is orthogonal) -- THE stationarity measure of the false vacuum,
    compared to FALSE_VAC_OFFSET_MAX x the vacuum separation by the
    FALSE-VACUUM guard of bounce_vgeneral_phi_v2 and by load_background.
    Aborts on a singular H'(0): no mass for a channel, not a nondegenerate
    minimum."""
    origin = np.zeros(pot_prime.N)
    try:
        step = np.linalg.solve(pot_prime.H(origin), pot_prime.dV(origin))
    except np.linalg.LinAlgError:
        raise RuntimeError(
            f'[ABORT FALSE-VACUUM] the Hessian at the false vacuum '
            f'{np.round(pot_prime.false_vac, 6)} is SINGULAR -- not a '
            f'nondegenerate minimum (a flat direction, no mass for at least '
            f'one channel).')
    return float(np.linalg.norm(step))


def channel_masses_and_mu(pot_prime):
    """THE channel masses^2 of the false vacuum -- PrimedPotential.m2, the
    eigenvalues numpy.linalg.eigh gave when the primed frame was built,
    heaviest first -- and the MSbar scale built from them, mu = sum_i m_i.
    Returns (m2, mu).
    At fixed couplings, lnD_ren depends on mu only through the ln mu^2 in
    its two finite add-backs, A1_fin and -A2_fin/2:
        mu d lnD_ren/dmu = -sum_i m_i^2 M_i/(8 pi^2) - (1/8) Int dr r^3 tr U_int^2
    with M_i = 2 pi^2 Int dr r^3 [U_int]_ii; with the couplings of S_E running
    with mu, the decay-rate exponent S_E + lnD_ren/2 is mu-independent to
    one-loop order.
    GUARD FV-MASS-POSITIVE (aborts): every m_i^2 must be positive -- m_i =
    sqrt(m_i^2) here and ln(m_i^2/mu^2) in the add-backs (tau_fin_fv, the
    two-mass bubble) would be NaN otherwise; a negative FV Hessian eigenvalue
    means the expansion point is not a minimum, and a zero one is a massless
    channel, which this calculation excludes.  The ONE check, where m2 is
    formed; every stage receives m2 from here."""
    m2 = pot_prime.m2.copy()
    if not np.all(m2 > 0.0):               # NaN-safe: catches NaN too
        raise RuntimeError(
            f'[ABORT FV-MASS-POSITIVE] non-positive FV channel mass^2 in '
            f'm2 = {m2}: the false vacuum is not a minimum or has a massless '
            f'channel (or the bounce npz is corrupt) -- mu = sum_i m_i and '
            f'ln(m_i^2/mu^2) would be NaN.')
    return m2, float(np.sum(np.sqrt(m2)))


def insertion(pot_prime, phi_b):
    """The interaction Hessian along a profile, U_int(r) = H'(phi_b(r)) -
    diag(m2): the Hessian of the cubic and quartic part of V', i.e. the
    field-dependent part of the primed Hessian, H'(phi_b) = diag(m2) + U_int
    (zero at the false vacuum); the insertion of the Born and Dyson series.
    ONE Hessian call on the whole profile; shape (nR, N, N) [radius, i, j]."""
    return pot_prime.H(phi_b) - np.diag(pot_prime.m2)


# The round-off level: an entry below this fraction of the largest one is
# round-off and counts as zero (coupled_pairs, the symmetry test of
# bounce_vgeneral_phi_v2, the component walls of the aside
# hessian_components_wall).
ROUNDOFF_REL = 1e-8


def coupled_pairs(U_int):
    """The channel pairs the insertion couples, (N, N) bool: True where the
    largest |U_int_ij| over r exceeds ROUNDOFF_REL of the largest entry of
    U_int.  A smaller entry is round-off of the Hessian and counts as zero
    (e.g. the OFF-DIAGONAL x-y entry of a spectator y the bounce leaves at
    zero; its diagonal U_yy, which the wall changes, still counts).  The ONE
    owner of this rule: bounce_polish fv_mass_min counts a channel for the
    solve radius when its row has a coupled pair (the channels the bounce
    acts on, nonzero U_int row), and GreensFunction builds its block table
    same_block from it.  U_int itself is never changed."""
    U_size = np.abs(U_int).max(axis=0)                 # (N, N): largest |U_ij| over r
    return U_size > ROUNDOFF_REL * U_size.max()


def trace_U_int_cubed(U_int):
    """tr U_int^3 (r) = sum_ijk [U_int]_ij [U_int]_jk [U_int]_ki on the grid
    (the trace of the matrix cube), whose radial moments set the high-s2
    tail of every band wave (pipeline_quadrature) and the asymptotic
    coefficients A3 and baacke_c (watchers_guards)."""
    return np.trace(U_int @ U_int @ U_int, axis1=1, axis2=2)


def wall_radius(R, phi_b):
    """r_W = R at max |dphi_b/dr| (vector norm over the N fields, frame-free)
    of a profile phi_b (nR, N) -- the ONE owner of the wall-radius rule."""
    R = np.asarray(R, float)
    dphi_b = np.gradient(np.asarray(phi_b, float), R, axis=0)
    speed = np.sqrt(np.sum(dphi_b * dphi_b, axis=1))
    return float(R[np.argmax(speed)])


# --------------------------------------------------------------------------- #
#  Stage-file I/O: the read rules of the stage npz and the band slices, the   #
#  fail-loud finiteness check, the crash-safe write                           #
# --------------------------------------------------------------------------- #
def load_stage_npz(path, bounce_npz, rerun_hint):
    """Open a stage npz and prove it belongs to THIS bounce: its stored bounce
    fingerprint (npz key `bounce_sha`) must equal
    bounce_fingerprint(bounce_npz).  A missing file, a missing stamp and a
    wrong stamp all abort, naming `rerun_hint` (the stage to run) -- the ONE
    read rule of the stage files."""
    if not os.path.isfile(path):
        raise RuntimeError(f'[ABORT] stage output missing: {path} -- run '
                           f'{rerun_hint} first.')
    stage_npz = np.load(path, allow_pickle=True)
    if 'bounce_sha' not in stage_npz.files:
        raise RuntimeError(
            f'[ABORT] {os.path.basename(path)} carries no bounce fingerprint '
            f'(npz key bounce_sha), so it cannot be shown to belong to this '
            f'bounce -- rerun {rerun_hint}.')
    cur_fingerprint = bounce_fingerprint(bounce_npz)
    if not cur_fingerprint:
        raise RuntimeError(
            f'[ABORT] cannot checksum the bounce npz {bounce_npz!r}, so '
            f'{os.path.basename(path)} cannot be checked against it -- pass '
            f'--bounce-npz (or set $VGENERAL_PHI_BOUNCE_NPZ) to the readable '
            f'bounce file.')
    if str(stage_npz['bounce_sha']) != cur_fingerprint:
        raise RuntimeError(
            f'[ABORT] {os.path.basename(path)} was computed against a '
            f'DIFFERENT bounce (fingerprint {str(stage_npz["bounce_sha"])[:8]} != '
            f'{cur_fingerprint[:8]}) -- rerun {rerun_hint} or point '
            f'--bounce-npz at the right file.')
    return stage_npz


def read_band_tables(data_dir, tag, *, expect_bounce_fingerprint, params,
                     fields, expect_tol=None, manifest_path=None, s2_cap=None):
    """Glob and merge every residual-band slice npz of this tag -- the base
    slices residual_band_<tag>_s2<v>.npz AND the wave-extension files
    residual_band_<tag>_s2<v>_n<lo>to<hi>.npz the adaptive driver writes --
    into one lookup  tables[s2][n] = delta_geq3, in ONE pass over the files.

    Fail-loud rules (no silent mixing):
      - s2_cap (if given): a slice with s2 above the cap is skipped -- it lies
        beyond the band rectangle (a resumed run whose watcher stopped below
        existing data) and never enters this read;
      - every kept slice carries the provenance stamps the band engine writes,
        and they must match this run:
          bounce_sha   = expect_bounce_fingerprint (a same-tag slice of
                         another bounce aborts);
          potential_id = potential_id(the slice's own m2, the RUN's
                         params/fields): the bounce fingerprint has already
                         pinned the model, so this catches a change of the
                         potential module's monomial basis ORDER since the
                         slice was integrated (the same coupling vector would
                         now mean another potential);
          code_version = VGENERAL_PHI_ENGINE_VERSION (the slice was written
                         by this engine build);
          rtol, atol   = expect_tol, if given (the same engine tolerances);
      - a wave present twice for the same s2 must agree to 1e-10 relative
        (else one of the files is stale -- abort);
      - manifest_path (if given): the kept set must equal the `slices` list
        of band_manifest_<tag>.json exactly (the MANIFEST guard -- a foreign
        same-tag slice would otherwise be summed in silently).
    Returns (s2_values sorted ascending, tables, file_list)."""
    import glob
    tables, files = {}, []
    for f in sorted(glob.glob(os.path.join(
            data_dir, f'residual_band_{tag}_s2*.npz'))):
        name = os.path.basename(f)
        with np.load(f, allow_pickle=True) as slice_npz:
            s2 = float(slice_npz['s2'])
            if s2_cap is not None and s2 > s2_cap:      # s2_cap is a node value
                continue
            files.append(f)
            missing = [k for k in ('bounce_sha', 'potential_id', 'm2',
                                   'code_version', 'rtol', 'atol')
                       if k not in slice_npz.files]
            if missing:
                raise RuntimeError(
                    f'[ABORT BAND-TABLES] {name} carries no {missing} stamp '
                    f'-- not a slice of the band engine; delete or retag it.')
            fingerprint = str(slice_npz['bounce_sha'])
            if fingerprint != expect_bounce_fingerprint:
                raise RuntimeError(
                    f'[ABORT BAND-TABLES] {name} was computed against a '
                    f'different bounce (fingerprint {fingerprint[:12]}... vs '
                    f'{expect_bounce_fingerprint[:12]}...) -- delete or '
                    f'retag it.')
            have_pid = str(slice_npz['potential_id'])
            want_pid = potential_id(np.asarray(slice_npz['m2'], float),
                                    params, fields)
            if have_pid != want_pid:
                raise RuntimeError(
                    f'[ABORT BAND-TABLES] {name} was integrated against a '
                    f'different potential:\n'
                    f'    slice stamp : {have_pid}\n'
                    f'    current     : {want_pid}\n'
                    f'The bounce fingerprint matched, so the model is the '
                    f'run\'s own: either the potential module\'s monomial basis '
                    f'ORDER changed since the slice was integrated (the '
                    f'coupling vector now means another potential) or the '
                    f'slice\'s m2 disagrees with its own stamp.  Regenerate '
                    f'the band.')
            engine_build = str(slice_npz['code_version'])
            if engine_build != VGENERAL_PHI_ENGINE_VERSION:
                raise RuntimeError(
                    f'[ABORT BAND-TABLES] {name} was written by band engine '
                    f'build {engine_build!r}, not the current '
                    f'{VGENERAL_PHI_ENGINE_VERSION!r} -- a stale same-tag slice '
                    f'from a materially different engine; delete or retag it.')
            if expect_tol is not None:
                slice_tol = (float(slice_npz['rtol']), float(slice_npz['atol']))
                if slice_tol != tuple(expect_tol):
                    raise RuntimeError(
                        f'[ABORT BAND-TABLES] {name} used engine tolerances '
                        f'{slice_tol} != expected {tuple(expect_tol)} -- delete '
                        f'or retag it.')
            traces_at_s2 = tables.setdefault(s2, {})
            for n, v in zip(np.asarray(slice_npz['n_values'], int),
                            np.asarray(slice_npz['delta_geq3'], float)):
                n = int(n)
                if n in traces_at_s2 and abs(v - traces_at_s2[n]) > 1e-10 * max(abs(v), 1e-30):
                    raise RuntimeError(
                        f'[ABORT BAND-TABLES] wave n={n} at s2={s2:g} appears '
                        f'in two slice files with conflicting values '
                        f'({traces_at_s2[n]!r} vs {v!r}) -- a stale extension '
                        f'file; delete it.')
                traces_at_s2[n] = float(v)
    if manifest_path:
        import json
        with open(manifest_path) as fh:
            want = set(os.path.basename(x) for x in json.load(fh)['slices'])
        have = set(os.path.basename(f) for f in files)
        extra, missing = sorted(have - want), sorted(want - have)
        if extra or missing:
            raise RuntimeError(
                f'[ABORT MANIFEST] band slice set does not match '
                f'{os.path.basename(manifest_path)}: EXTRA (foreign) {extra}; '
                f'MISSING {missing}.  Remove the foreign slice(s) or '
                f'regenerate the band.')
    return np.array(sorted(tables.keys())), tables, files


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
    convention for the stage codes; the driver spells the same names in
    shell)."""
    return {
        'bounce':         os.path.join(data_dir, f'bounce_data_{tag}.npz'),
        'band_cutoffs':   os.path.join(data_dir, f'band_cutoffs_{tag}.npz'),
        'eig_n0':         os.path.join(data_dir, f'eig_n0_{tag}.npz'),
        'eig_n1':         os.path.join(data_dir, f'eig_n1_{tag}.npz'),
        'sector_n0':      os.path.join(data_dir, f'sector_n0_{tag}.npz'),
        'sector_n1':      os.path.join(data_dir, f'sector_n1_{tag}.npz'),
        'band_integrals': os.path.join(data_dir, f'band_integrals_{tag}.npz'),
        'tail_high_n':    os.path.join(data_dir, f'tail_high_n_{tag}.npz'),
        'addback_tadpole': os.path.join(data_dir,
                                        f'addback_tadpole_{tag}.npz'),
        'addback_fish':    os.path.join(data_dir, f'addback_fish_{tag}.npz'),
        # the final result file IS lnD_ren
        'lnD_ren':        os.path.join(data_dir, f'lnD_ren_{tag}.npz'),
    }


def add_standard_cli(parser):
    """The uniform CLI prefix of every stage file:
    --data-dir --bounce-npz --tag (stage-specific flags follow)."""
    parser.add_argument('--data-dir',
                        default=os.environ.get('VGENERAL_PHI_DATA', '.'),
                        help='data directory (all stage npz live here; '
                             'default $VGENERAL_PHI_DATA)')
    parser.add_argument('--bounce-npz',
                        default=os.environ.get('VGENERAL_PHI_BOUNCE_NPZ'),
                        help='the bounce npz of stage 0 (bounce_data_<tag>.npz, '
                             'read by load_background; default '
                             '$VGENERAL_PHI_BOUNCE_NPZ)')
    parser.add_argument('--tag', default='vgeneral_phi_v2',
                        help='run tag, threaded into every filename (the '
                             'driver requires "vgeneral_phi_v2" in it; the '
                             'plot codes find a run by it)')
    return parser


def bounce_fingerprint(path):
    """The FINGERPRINT of a bounce file: the SHA-256 checksum of the npz
    bytes, a 64-character hex string that changes if a single byte of the
    file changes ('' when there is no such file).  It identifies the exact
    bounce file a stage output was computed from (provenance, never physics);
    every stage stores it under the npz key `bounce_sha`."""
    if not path or not os.path.isfile(path):
        return ''
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def potential_id(m2, params, fields):
    """Short string identifying the potential a stage output belongs to:
    round(m^2,6) of every channel + `phys`, a sha256 prefix of the coupling
    vector (rounded to 12 digits) and the monomial basis ORDER; `params`/
    `fields` are the RUN's own model (background dict / bounce npz).
    `phys` hashes CONTENT, never file bytes: a comment edit in the potential
    module must not invalidate a finished band, and exact floats/strings
    hash the same on every machine.  Provenance metadata (not physics)."""
    import potential_vgeneral_phi_v2 as potential_mod
    content = (
        tuple(round(float(v), 12) for v in np.asarray(params, float)),
        tuple(potential_mod.monomial_order(tuple(fields))),
    )
    couplings_hash = hashlib.sha256(repr(content).encode()).hexdigest()[:12]
    m2a = np.asarray(m2, float).ravel()
    m2s = ','.join(str(round(float(v), 6)) for v in m2a)
    return f"m2={m2s}|phys={couplings_hash}"


def provenance_stamp(bounce_npz, m2, params, fields):
    """The three metadata keys every stage output npz carries: the bounce
    fingerprint, the potential_id of the run's own model (m2, params, fields
    from the background dict load_background built from the bounce npz), and
    the pipeline version."""
    return dict(bounce_sha=bounce_fingerprint(bounce_npz),
                potential_id=potential_id(m2, params, fields),
                code_version=VGENERAL_PHI_PIPELINE_VERSION)


def s2_filetag(s2):
    """The s^2 filename tag of every s2-parametrised npz and figure: 6
    decimals, '.' -> 'p' (0.5 -> '0p500000').  The band slices are the one
    exception: their names carry band_s2_text below, because the band
    ladder reaches down to the model's own lowest s2 scale."""
    return f'{float(s2):.6f}'.replace('.', 'p')


def band_s2_text(s2):
    """The text of one band s2 node, 7 significant digits (format '.6e',
    precise at every scale): the one spelling the band ladder rounds its
    nodes to (band_adaptive.s2_ladder), the engine command line carries
    (band_adaptive.engine_cmd) and the slice file name carries, '.' -> 'p'
    (greens_function._run_slice)."""
    return f'{float(s2):.6e}'


def bounce_name(bounce_data):
    """The bounce NAME <falseLabel>_<trueLabel> of an OPEN bounce npz
    `bounce_data` (or its plain dict): the stored pair_name without its
    parenthetical file note, spaces as underscores -- the name every figure
    file and picture folder carries.  A bounce npz of an older stage 0 names
    an intermediate minimum at an end of the bounce MFV<i> / MTV<i>; it is
    read with today's label M<i> (bounce_vgeneral_phi_v2.vacuum_label), so
    every picture of every run carries one label per vacuum.  Aborts on an
    npz without pair_name."""
    if 'pair_name' not in bounce_data:
        raise RuntimeError('[ABORT] the bounce npz stores no pair_name key -- '
                           'not a bounce written by bounce_vgeneral_phi_v2.')
    name = str(bounce_data['pair_name']).split('(')[0].strip().replace(' ', '_')
    return re.sub(r'M[FT]V(\d+)', r'M\1', name)


# --------------------------------------------------------------------------- #
#  Index bookkeeping of the Green's-function assembly                         #
# --------------------------------------------------------------------------- #
# Every stack is stored [radius, component i, basis function a] (a branch h^+
# or h^- on the radial grid).  The constant Wronskian of wave n is
#     C[a, b] = delta_ab + h^-[a, b](inf)
#             = delta_ab + h^-[a, b](r_max)
#               + r_max I_nu K_nu(kappa_a r_max) h^-'[a, b](r_max)
#     (equal because U_int = 0 beyond r_max; engine.h_minus_inf): first
#     slot a = the '+' basis function (at r_max, where h^+ = 0, the unit vector
#     of component a), second slot b = the '-' basis function.  With
#     nu = n + 1, kappa_i = sqrt(m_i^2 + s^2), B^+_i(r) = K_nu(kappa_i r)/r and
#     B^-_i(r) = I_nu(kappa_i r)/r the radial Green's function of wave n is
#     r > r':  [G_n(r, r')]_ik = B^+_i(r) B^-_k(r') sum_ab [1+h^+(r)]_ia [C^-1]_ba [1+h^-(r')]_kb
#     r < r':  [G_n(r, r')]_ik = B^-_i(r) B^+_k(r') sum_ab [1+h^-(r)]_ia [C^-1]_ab [1+h^+(r')]_kb
# In BOTH terms the '-' label sits in the FIRST slot of C^-1 and the '+'
# label in the SECOND (C^-1 inverts C[+, -]).  The two sums are the brackets
# below: (1+h^+) C^-T (1+h^-)^T, einsum letters 'ba' (bracket_outer_pairs;
# at r = r' bracket_outer, the coincident bracket G_bracket), and
# (1+h^-) C^-1 (1+h^+)^T, letters 'ab' (bracket_inner_pairs).

def radius_first(y):
    """(..., n_r) solver block -> (n_r, ...) stack: the radius axis moved to
    the front.  Array layout only; no physics index is touched."""
    return np.ascontiguousarray(np.moveaxis(y, -1, 0))


def bracket_outer(plus_stack, C_inv, minus_stack):
    """[G_bracket]_ik = sum_ab plus_stack[i,a] C_inv[b,a] minus_stack[k,b],
    both stacks at the SAME radius: the coincident bracket
    (1+h^+) C^-T (1+h^-)^T, the r > r' sum of the section comment above at
    r = r', so that [G_n(r, r)]_ik = B^+_i(r) B^-_k(r) [G_bracket]_ik.
    The basis-function label of the '-' branch (b) sits in the FIRST slot of
    C_inv, that of the '+' branch (a) in the SECOND.
    plus_stack, minus_stack: (n_r, N, N) stacks [radius, component, basis
          function] or (N, N) constants;
    C_inv: (N, N) the inverse of the constant Wronskian C.
    Result: (n_r, N, N) or (N, N); '...' is the radius axis, absent for a
    constant."""
    return np.einsum('...ia,ba,...kb->...ik', plus_stack, C_inv, minus_stack)


def bracket_outer_pairs(plus_stack, C_inv, minus_stack):
    """[bracket]_{r r' i k} = sum_{a,b} plus_stack[r,i,a] * C_inv[b,a] *
    minus_stack[r',k,b]  -- the outer bracket on two radial grids (the r > r'
    triangle).  plus_stack: (n_r, N, N), minus_stack: (n_r', N, N); result
    (n_r, n_r', N, N); r and s are the two radius axes."""
    return np.einsum('ria,ba,skb->rsik', plus_stack, C_inv, minus_stack)


def bracket_inner_pairs(minus_stack, C_inv, plus_stack):
    """[bracket]_{r r' i k} = sum_{a,b} minus_stack[r,i,a] * C_inv[a,b] *
    plus_stack[r',k,b]  -- the inner bracket on two radial grids (the r < r'
    triangle); C_inv in its natural slot order."""
    return np.einsum('ria,ab,skb->rsik', minus_stack, C_inv, plus_stack)
