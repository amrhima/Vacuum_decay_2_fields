#!/usr/bin/env bash
# run_vgeneral_phi_v2_production.sh -- the production driver: lnD_ren of one
# bounce of a polynomial potential of total degree <= 4 in N scalar fields
# (N is set by --fields and never hardcoded: every matrix of the fluctuation
# problem is N x N).  The stages run in the order below; each is one code of
# computation/, reads the files of the stages above it and writes its own
# <name>_<tag>.npz into $DATA_DIR.
#
#   0  bounce           the vacua of the potential (labels TV0, M<i>, FV0);
#                       for every downward pair a CosmoTransitions seed, the
#                       solve_bvp solve of the O(4) equation of motion (it
#                       carries all the accuracy), the gates on the final
#                       profile, the mode census of the waves n = 0, 1 (the s2
#                       where det C = 0) and a reference profile at a tighter
#                       tolerance; the gallery; then the SELECTION GATE: only
#                       a bounce with one negative mode (n = 0) and one
#                       translation zero mode (n = 1; positive n = 1 bound
#                       states are allowed) goes on
#                       (bounce_vgeneral_phi_v2.py -> bounce_data)
#   0a eigenvalue n=0   the negative-mode eigenvalue omega2_00, read from the
#                       census (eigenvalue_neg_vgeneral_phi_v2.py -> eig_n0)
#   0b eigenvalue n=1   the zero-mode eigenvalue omega2_10, read from the
#                       census; guard ZERO-MODE
#                       (eigenvalue_zero_vgeneral_phi_v2.py -> eig_n1).  0a and
#                       0b repeat the selection gate, so the band, the longest
#                       stage, never starts on a bounce the assembly is not
#                       written for
#   1  band             delta_geq3(n, s2), the trace of the Green's function
#                       of wave n with Born orders 0, 1 and 2 removed, for
#                       n = 2..n_max and 0 < s2 <= s2_max, in rounds of
#                       parallel engine workers; the watchers fix n_max,
#                       s2_max and fit_lo (band_adaptive_vgeneral_phi_v2.py
#                       -> the residual_band slices, band_cutoffs,
#                       band_manifest)
#   2a sector n=0       Delta_n = -Int_0^s2_max [delta_geq3
#                       - 1/(omega2_00 + s2)] ds2 of the wave n = 0 on its own
#                       s2 grid (gbar_sector_vgeneral_phi_v2.py --sector-n 0
#                       -> sector_n0)
#   2b sector n=1       the same for n = 1 with the pole 4/(omega2_10 + s2)
#                       (gbar_sector_vgeneral_phi_v2.py --sector-n 1
#                       -> sector_n1)
#   3  band integrals   per wave n = 2..n_max: I_n(s2_max) =
#                       -Int_0^s2_max delta_geq3 ds2, the high-s2 tail
#                       T_n(s2_max) = -Int_s2_max^inf delta_geq3 ds2 (from the
#                       large-s2 shape of the wave) and their sum I_n(inf)
#                       (band_integrals_vgeneral_phi_v2.py -> band_integrals)
#   4  high-n tail      one fit a/nu^3 + b/nu^5 + c/nu^7 of I_n(inf) on
#                       [fit_lo, n_max] and its Hurwitz-zeta sum Zcal over
#                       n > n_max (tail_high_n_zeta_vgeneral_phi_v2.py
#                       -> tail_high_n)
#   5  tadpole add-back A1_fin = sum_i M_i tau_fin_fv(m_i^2), the MS-bar
#                       finite part of Born order 1, traced and integrated
#                       over s2 (addback_tadpole_vgeneral_phi_v2.py
#                       -> addback_tadpole)
#   6  fish add-back    A2_fin, the MS-bar finite part of Born order 2; it
#                       enters lnD_ren as -A2_fin/2
#                       (addback_fish_vgeneral_phi_v2.py -> addback_fish)
#   7  assembly         lnD_ren = (T1) the sectors 2a, 2b with the
#                       fictitious-mode block and ln|omega2_00|
#                       + (T2) + (T4) the Born-order-1 add-back (= A1_fin)
#                       + (T3) sum_n I_n(s2_max) + (T5) sum_n T_n(s2_max)
#                       + (T6) Zcal + (T7) -A2_fin/2; and the bounce
#                       sensitivity indicator eps_bounce
#                       (assemble_lnD_vgeneral_phi_v2.py -> lnD_ren)
# Stages 5 and 6 need only the bounce: they could run at any point after
# stage 0.
#
# WATCHERS AND GUARDS.  Stage 1 grows the band in rounds until its watchers
# find the large-s2 and large-n behaviour of THIS bounce:
#   S2-PLATEAU     the raw band sum converges in its cutoff like
#                  baacke_c/(s2_max + m_min2), and the predicted error of the
#                  completed sum is <= eps_s2                         -> s2_max
#   SECTOR-CUTOFF  the part of lnD_ren the sectors n = 0, 1 drop above
#                  s2_max (extrapolated) is <= SECTOR_REMAINDER_TOL   -> s2_max
#   N-ONSET        nu^3 I_n(inf) stays close to the analytic A3       -> fit_lo
#   N-TAIL-BOUND   the fitted subleading tail is <= eps_n and the
#                  fitted a agrees with A3                            -> n_max
# Later stages re-check the same conditions as guards (S2-BAACKE,
# N-ASYMPTOTE, N-TAIL-BOUND, SECTOR-CUTOFF).  Every guard stops the run with
# [ABORT <NAME>], explained in the file that raises it.  Nothing a watcher or
# guard computes enters lnD_ren (one exception: the SHAPE-RATIO repair of
# stage 3), and no cutoff is set by hand.
#
# RESUME.  A stage is skipped when its output (named with $TAG) exists, is not
# older than the files it is computed from and stores the cutoffs and
# tolerances of this run; stage 1 re-reads its slices and computes only the
# missing ones.  So a rerun of the same command continues where the last one
# stopped, and never uses another run's data.
#
# REGRESSION PINS (optional): PIN_NMAX and PIN_S2MAX (set together) fix the
# band rectangle, so the watchers measure but do not extend it; PIN_FITLO
# fixes the start of the fit window, with or without them.  Every guard still
# checks.
set -e
set -o pipefail

# no .pyc litter in computation/ (module-level flags cannot protect ad-hoc
# imports; the environment variable protects everything launched from here)
export PYTHONDONTWRITEBYTECODE=1
# avoid BLAS oversubscription when BAND_JOBS>1 (one BLAS thread per process)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# a relative PY (e.g. env/bin/python) names a file relative to the caller's
# folder: make it absolute before the cd below
case "${PY:-}" in */*) [ -e "$PY" ] && PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")";; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
COMPUTATION_DIR=computation

# ================== TOP-LEVEL CLI (the user-facing interface) ================
# Every option but --help sets one shell variable -- FIELDS_ARG,
# COUPLINGS_ARG, PAIR, SELECT_BOUNCE, SEEDRANGE_ARG, DATA_DIR, TAG -- which may
# equally come from the environment (FIELDS_ARG='--fields x,y' carries its
# flag); the CLI wins when both are given.  Unknown options FAIL loud.
usage(){
  cat <<'EOF'
Usage: ./run_vgeneral_phi_v2_production.sh [options]

  Set the potential, pick (or interactively choose) a bounce, and compute
  ln D_ren for it.  The potential is REQUIRED: pass --fields and --couplings
  (worked example below); a run without them aborts.

Options (write the values of --fields, --couplings and --seed-range without spaces):
  --fields x,y,z          the scalar fields (distinct single lowercase letters)
  --couplings 'x2=-2,...' the potential: degree-<=4 monomial coefficients
                          (names concatenate field+power: x2 = x^2, xy2 = x*y^2)
  --pair NAME             which (false,true) vacuum pair to run; both
                          spellings work: the label name <falseLabel>_<trueLabel>
                          (FV0_TV0, M1_TV0, FV0_M2, ... -- TV0 = the deepest
                          minimum, M1, M2, ... the next ones upwards, FV0 = the
                          highest) or the raw file form F<i>_T<j> (list them:
                          python3 computation/bounce_vgeneral_phi_v2.py
                          --fields ... --couplings ... --list-bounces)
  --select-bounce         solve every downward pair, plot the bounce gallery,
                          and ASK which bounce to use (needs a terminal)
  --seed-range LO,HI      vacuum-search seed box (default: derived from the
                          couplings -- the Cauchy root bound of grad V)
  --data-dir DIR          where data/logs land (default: ./results next to
                          the pipeline; or set the DATA_DIR env var)
  --tag TAG               explicit data tag, must contain vgeneral_phi_v2
                          (default: auto, unique per pair+potential)
  -h, --help              this help

Example (three coupled fields, interactive bounce choice):
  ./run_vgeneral_phi_v2_production.sh --fields x,y,z \
      --couplings 'c=1.2,x2=-2,y2=-2,xy2=1,x2y2=0.15,x4=1,y4=1,z2=0.5,x2z=0.2,z4=0.25' \
      --select-bounce --data-dir results_xyz
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --fields)        [ $# -ge 2 ] || { echo "[ABORT] --fields needs a value"; exit 1; }
                     FIELDS_ARG="--fields $2"; shift 2;;
    --couplings)     [ $# -ge 2 ] || { echo "[ABORT] --couplings needs a value"; exit 1; }
                     COUPLINGS_ARG="--couplings $2"; shift 2;;
    --pair)          [ $# -ge 2 ] || { echo "[ABORT] --pair needs a value"; exit 1; }
                     PAIR="$2"; shift 2;;
    --select-bounce) SELECT_BOUNCE=1; shift;;
    --seed-range)    [ $# -ge 2 ] || { echo "[ABORT] --seed-range needs LO,HI"; exit 1; }
                     SEEDRANGE_ARG="--seed-range=$2"; shift 2;;   # = form: a leading-dash LO (-1,15) survives argparse
    --data-dir)      [ $# -ge 2 ] || { echo "[ABORT] --data-dir needs a value"; exit 1; }
                     DATA_DIR="$2"; shift 2;;
    --tag)           [ $# -ge 2 ] || { echo "[ABORT] --tag needs a value"; exit 1; }
                     TAG="$2"; shift 2;;
    *) echo "[ABORT] unknown option: $1  (see --help)"; exit 1;;
  esac
done

# ---- the interpreter: PY=<python> or python3, which must import the stack
#      (always written "$PY": a path with a space stays one word)
PY="${PY:-python3}"
if ! "$PY" -c 'import numpy, scipy.integrate, sympy, matplotlib, cosmoTransitions' >/dev/null 2>&1; then
  echo "[ABORT] python interpreter unusable: $PY (an import of numpy, scipy.integrate, sympy, matplotlib or cosmoTransitions failed)." >&2
  echo "        Install the requirements (pip install -r $SCRIPT_DIR/requirements.txt)" >&2
  echo "        or point PY=<python> at an interpreter that has them." >&2
  exit 1
fi
echo "[python] $PY ($("$PY" -V 2>&1))"

# ---- storage: a RELATIVE --data-dir is resolved against the caller-visible
#      pipeline root; the default is a local ./results next to the pipeline
#      (override with --data-dir or the DATA_DIR env var) ----
if [ -z "${DATA_DIR:-}" ]; then
  DATA_DIR="$SCRIPT_DIR/results"
fi
case "$DATA_DIR" in /*) ;; *) DATA_DIR="$SCRIPT_DIR/$DATA_DIR";; esac
mkdir -p "$DATA_DIR"
# the model is REQUIRED: this pipeline ships NO built-in potential.
if [ -z "${COUPLINGS_ARG:-}" ] || [ -z "${FIELDS_ARG:-}" ]; then
  echo "[ABORT] no potential given: this pipeline ships NO built-in model."
  echo "        Pass --fields AND --couplings, e.g."
  echo "        ./run_vgeneral_phi_v2_production.sh --fields x,y \\"
  echo "            --couplings 'c=1.2,x2=-2,y2=-2,xy2=1,x2y2=0.15,x4=1,y4=1' --select-bounce"
  exit 2
fi
# ... and so is the bounce choice: --select-bounce (type the name at the
# gallery prompt) or --pair NAME.  There is no default pair.
if [ -z "${PAIR:-}" ] && [ "${SELECT_BOUNCE:-0}" != "1" ]; then
  echo "[ABORT] no bounce chosen for the potential."
  echo "        Add  --select-bounce   to see every named bounce and type your choice,"
  echo "        or   --pair <NAME>     if you already know it (names: --list-bounces)."
  exit 2
fi
# an explicit --tag must contain vgeneral_phi_v2 (every stage file name carries the
# tag, and the plot codes find a run by it); checked before anything is solved.
# The automatic tag below always contains it.
case "${TAG:-vgeneral_phi_v2}" in *vgeneral_phi_v2*) ;; *) echo "[ABORT] TAG='$TAG' must contain 'vgeneral_phi_v2'"; exit 1;; esac

# ---- INTERACTIVE bounce selection runs FIRST (before the tag exists) -------
# --select-bounce: the bounce code of stage 0 solves every downward vacuum
# pair, draws the gallery into $DATA_DIR/gallery (per bounce
# bounce_<name>_profiles.png, the profiles with S_E in the legend, and for
# two or three fields bounce_<name>_potential.png, the path with the
# potential), prints the menu of names and S_E and asks for a bounce by
# name; this needs a terminal.  The choice
# becomes PAIR and the run goes on exactly like a --pair run of that pair,
# under that pair's own data tag.  The solved pairs stay in
# $DATA_DIR/bounce_gen, and stage 0 reuses them.
if [ "${SELECT_BOUNCE:-0}" = "1" ]; then
  _SEL_TMP="$DATA_DIR/bounce_data_seltmp_vgeneral_phi_v2.npz"
  rm -f "$_SEL_TMP"
  "$PY" $COMPUTATION_DIR/bounce_vgeneral_phi_v2.py --data-dir "$DATA_DIR" --tag "seltmp_vgeneral_phi_v2" \
      --select-bounce ${FIELDS_ARG:-} ${COUPLINGS_ARG:-} ${SEEDRANGE_ARG:-}
  [ -f "$_SEL_TMP" ] || { echo "[ABORT] bounce selection did not complete (no terminal? rerun with --pair)"; exit 1; }
  PAIR=$("$PY" -c "import sys, numpy as np; d=np.load(sys.argv[1], allow_pickle=True); print(f\"F{int(d['false_index'])}_T{int(d['true_index'])}\")" "$_SEL_TMP")
  rm -f "$_SEL_TMP"
  echo "[select] chosen bounce: $PAIR -- continuing as a --pair $PAIR run"
fi

# ---- automatic data tag (no --tag given) -----------------------------------
#   rk_vgeneral_phi_v2_<NAME>_<Nfields>_<Ncouplings>_<date>_<time>
# NAME is the bounce name in the vacuum labels (TV0, M1, M2, ..., FV0), however
# --pair was written (a name or the file form F<i>_T<j>; --select-bounce gives
# the file form): stage 0 resolves it in the vacuum catalogue (--pair-name, the
# last line it prints) and stops the run on a pair that does not exist.  The
# pair is part of the tag, so two pairs of one potential never share a bounce
# file; the tag is stored per (bounce name, potential) in autotag_<hash>.txt, so
# a resumed run finds the same tag.  An explicit --tag wins.
# portable content hash for the autotag key (macOS: md5, Linux: md5sum)
md5_10(){
  if command -v md5 >/dev/null 2>&1; then md5; else md5sum | awk '{print $1}'; fi | cut -c1-10
}
if [ -z "${TAG:-}" ]; then
  _PAIR_OUT=$("$PY" $COMPUTATION_DIR/bounce_vgeneral_phi_v2.py --pair "$PAIR" --pair-name \
      ${FIELDS_ARG:-} ${COUPLINGS_ARG:-} ${SEEDRANGE_ARG:-}) || exit 1
  PAIR=$(printf '%s\n' "$_PAIR_OUT" | tail -n 1)
  _COUP_SPEC=$(echo "${COUPLINGS_ARG#*--couplings}" | tr -d " '\"")
  _FIELD_SPEC=$(echo "${FIELDS_ARG#*--fields}" | tr -d " '\"")
  N_FIELDS=$(echo "$_FIELD_SPEC" | awk -F',' '{print NF}')
  N_COUPS=$(echo "$_COUP_SPEC" | awk -F',' '{n=0; for(i=1;i<=NF;i++) if($i!="") n++; print n}')
  _AUTOKEY=$(printf '%s|%s|%s' "$PAIR" "$_FIELD_SPEC" "$_COUP_SPEC" | md5_10)
  _TAGFILE="$DATA_DIR/autotag_${_AUTOKEY}.txt"
  if [ -f "$_TAGFILE" ]; then
    TAG=$(cat "$_TAGFILE")
    echo "[autotag] same potential as a previous run -> reusing persisted tag $TAG (resume-safe)"
  else
    TAG="rk_vgeneral_phi_v2_${PAIR}_${N_FIELDS}_${N_COUPS}_$(date '+%Y%m%d_%H%M%S')"
    echo "$TAG" > "$_TAGFILE"
    echo "[autotag] new data tag $TAG (pair=$PAIR fields=$N_FIELDS couplings=$N_COUPS)"
  fi
fi
BOUNCE="$DATA_DIR/bounce_data_${TAG}.npz"
LOG="$DATA_DIR/run.log"

# ---- the run settings.  The accuracy parameters are dimensionless, the
#      same for every model: EPS_S2 and EPS_N are the targets of two error
#      estimates of lnD_ren (the drift the high-s2 tail leaves in the completed
#      band sum; the fitted subleading part of the high-n tail);
#      ONSET_TOL and A3_TOL are relative deviations from the analytic A3;
#      S2_STEP_TOL is the change of each band integral I_n, relative to the
#      band's largest |I_n|, when every second s2 node is dropped (stage 3);
#      SECTOR_STEP_TOL is the absolute
#      change of the sector term Delta_n of lnD_ren under the same
#      halving (stages 2a/2b double their nodes until it passes).  Every
#      model-dependent cutoff, fit window and scan window is found by the
#      watchers or derived from the bounce -- none of them is set here. ----
EPS_S2=${EPS_S2:-1e-2}           # S2-PLATEAU/S2-BAACKE target
EPS_N=${EPS_N:-1e-2}             # N-TAIL-BOUND target
ONSET_TOL=${ONSET_TOL:-0.2}      # N-ONSET sustained-match tolerance
A3_TOL=${A3_TOL:-0.10}           # N-ASYMPTOTE tolerance
BAND_JOBS=${BAND_JOBS:-2}        # concurrent band-engine workers
S2_STEP_TOL=${S2_STEP_TOL:-0.005}         # S2-STEP limit (stage 3)
SECTOR_STEP_TOL=${SECTOR_STEP_TOL:-1e-3}  # SECTOR-STEP limit (stages 2a/2b)
# SECTOR_S2_LO (empty by default) is the one opt-in override that carries
# units (an absolute s2 = mass^2): empty, the sector stages derive the low end
# of their s2 grid from the model (pipeline_quadrature sector_s2_lo_default),
# never above the band's lowest node s2_min; set it only to reproduce a
# stored run bit-exactly, alongside the PIN_* regression pins.  When set, the
# flag is passed and the sector skip-guards value-match it.
SECTOR_S2_LO=${SECTOR_S2_LO:-}
SECTOR_S2_LO_FLAG=${SECTOR_S2_LO:+--sector-s2-lo $SECTOR_S2_LO}
SECTOR_S2_LO_MATCH="sector_s2_lo_is_default 1"
if [ -n "$SECTOR_S2_LO" ]; then
  SECTOR_S2_LO_MATCH="sector_s2_lo_is_default 0 sector_s2_lo $SECTOR_S2_LO"
fi
# optional regression pins: PIN_NMAX and PIN_S2MAX together, PIN_FITLO with
# or without them (see the header)
PIN_NMAX=${PIN_NMAX:-}
PIN_S2MAX=${PIN_S2MAX:-}
PIN_FITLO=${PIN_FITLO:-}

# per-stage output npz (one file per physics piece; stage skip-guards)
BAND_CUTOFFS_NPZ="$DATA_DIR/band_cutoffs_${TAG}.npz"
MANIFEST="$DATA_DIR/band_manifest_${TAG}.json"
EIG_N0_NPZ="$DATA_DIR/eig_n0_${TAG}.npz"
EIG_N1_NPZ="$DATA_DIR/eig_n1_${TAG}.npz"
SECTOR_N0_NPZ="$DATA_DIR/sector_n0_${TAG}.npz"
SECTOR_N1_NPZ="$DATA_DIR/sector_n1_${TAG}.npz"
BAND_INTEGRALS_NPZ="$DATA_DIR/band_integrals_${TAG}.npz"
TAIL_HIGH_N_NPZ="$DATA_DIR/tail_high_n_${TAG}.npz"
ADDBACK_TADPOLE_NPZ="$DATA_DIR/addback_tadpole_${TAG}.npz"
ADDBACK_FISH_NPZ="$DATA_DIR/addback_fish_${TAG}.npz"
LND_REN_NPZ="$DATA_DIR/lnD_ren_${TAG}.npz"

ts(){ date '+%H:%M:%S'; }
stage(){ echo "" | tee -a "$LOG"; echo "===== [$(ts)] $* =====" | tee -a "$LOG"; }
# Run a stage command: timing, tee into the run log, loud abort on any
# non-zero exit -- including the census abort of a SELECTED non-index-1
# bounce (stage 0's selection gate; stages 0a/0b are its backstops).
# usage: run_stage <label> <cmd...>
run_stage(){
  local label="$1"; shift
  set +e
  time "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" -eq 0 ] || { echo "[ABORT] stage ($label) failed (exit $rc)" | tee -a "$LOG"; exit "$rc"; }
}
# value-matched skip-guard: a stage output is reused ONLY if the npz stores
# the SAME cutoff values this run derived (a resumed dir whose earlier run
# chose different cutoffs must recompute, never silently reuse).
# usage: npz_matches FILE key1 val1 [key2 val2 ...] -> exit 0 iff all match
npz_matches(){
  local f="$1"; shift
  [ -f "$f" ] || return 1
  NPZF="$f" NPZARGS="$*" "$PY" - <<'PYEOF'
import os, sys
import numpy as np
d = np.load(os.environ['NPZF'], allow_pickle=True)
a = os.environ['NPZARGS'].split()
for k, v in zip(a[::2], a[1::2]):
    # np.isclose is False when either value is NaN: a NaN never matches
    if k not in d.files or not np.isclose(float(d[k]), float(v), rtol=1e-12, atol=0.0):
        sys.exit(1)
sys.exit(0)
PYEOF
}
# append (-a): a resumed run must not erase the previous run's log
echo "[$(ts)] vgeneral_phi_v2 PRODUCTION run: tag=$TAG data=$DATA_DIR" | tee -a "$LOG"
echo "[$(ts)] accuracy targets: eps_s2=$EPS_S2 eps_n=$EPS_N onset_tol=$ONSET_TOL a3_tol=$A3_TOL; ${BAND_JOBS}-way band" | tee -a "$LOG"

stage "0: bounce -> $TAG (every downward pair: seed -> solve -> gates -> mode census; gallery; selection gate)"
run_stage "0 bounce" "$PY" $COMPUTATION_DIR/bounce_vgeneral_phi_v2.py --data-dir "$DATA_DIR" --tag "$TAG" \
     --pair "$PAIR" ${FIELDS_ARG:-} ${COUPLINGS_ARG:-} ${SEEDRANGE_ARG:-}
[ -f "$BOUNCE" ] || { echo "[ABORT] stage 0 did not produce $BOUNCE" | tee -a "$LOG"; exit 1; }
echo "  bounce: $BOUNCE" | tee -a "$LOG"

# ---- STAGES 0a/0b run BEFORE the band: they need only the bounce's stored
#      census and repeat the selection gate before the longest stage starts;
#      the eigenvalues they write feed the sectors 2a/2b after the band. ----
stage "0a: EIGENVALUE n=0 -> $EIG_N0_NPZ (the negative-mode eigenvalue omega2_00 read from the stage-0 census; selection-gate backstop before the band)"
# The eig npz depends on the bounce alone: resume on existence + not older
# than the bounce.
if [ -f "$EIG_N0_NPZ" ] && [ ! "$BOUNCE" -nt "$EIG_N0_NPZ" ]; then
  echo "  [skip] $EIG_N0_NPZ exists and is newer than the bounce" | tee -a "$LOG"
else
  run_stage "0a eigenvalue n=0" "$PY" $COMPUTATION_DIR/eigenvalue_neg_vgeneral_phi_v2.py --data-dir "$DATA_DIR" \
       --bounce-npz "$BOUNCE" --tag "$TAG"
  [ -f "$EIG_N0_NPZ" ] || { echo "[ABORT] stage 0a did not write $EIG_N0_NPZ" | tee -a "$LOG"; exit 1; }
fi

stage "0b: EIGENVALUE n=1 -> $EIG_N1_NPZ (the zero-mode eigenvalue omega2_10 read from the stage-0 census, ZERO-MODE guard; aborts on more than one dangerous n=1 crossing before the band)"
if [ -f "$EIG_N1_NPZ" ] && [ ! "$BOUNCE" -nt "$EIG_N1_NPZ" ]; then
  echo "  [skip] $EIG_N1_NPZ exists and is newer than the bounce" | tee -a "$LOG"
else
  run_stage "0b eigenvalue n=1" "$PY" $COMPUTATION_DIR/eigenvalue_zero_vgeneral_phi_v2.py --data-dir "$DATA_DIR" \
       --bounce-npz "$BOUNCE" --tag "$TAG"
  [ -f "$EIG_N1_NPZ" ] || { echo "[ABORT] stage 0b did not write $EIG_N1_NPZ" | tee -a "$LOG"; exit 1; }
fi

stage "1: ADAPTIVE BAND DRIVER (watchers find n_max, s2_max, fit_lo; ${BAND_JOBS} workers)"
# The driver re-reads existing slices (provenance-checked) and computes only
# what is missing, so re-running it on a complete band is cheap -- it is its
# own skip-guard and always re-writes band_cutoffs + the manifest.
PIN_FLAGS=""
if [ -n "$PIN_NMAX" ] || [ -n "$PIN_S2MAX" ]; then
  if [ -z "$PIN_NMAX" ] || [ -z "$PIN_S2MAX" ]; then
    echo "[ABORT] PIN_NMAX and PIN_S2MAX must be set together (got PIN_NMAX='$PIN_NMAX' PIN_S2MAX='$PIN_S2MAX')" | tee -a "$LOG"
    exit 1
  fi
  PIN_FLAGS="--pin-n-max $PIN_NMAX --pin-s2-max $PIN_S2MAX"
  echo "  [PIN] regression rectangle: n_max=$PIN_NMAX s2_max=$PIN_S2MAX" | tee -a "$LOG"
fi
if [ -n "$PIN_FITLO" ]; then
  PIN_FLAGS="$PIN_FLAGS --pin-fit-lo $PIN_FITLO"
  echo "  [PIN] fit-window start: fit_lo=$PIN_FITLO" | tee -a "$LOG"
fi
run_stage "1 band driver" "$PY" $COMPUTATION_DIR/band_adaptive_vgeneral_phi_v2.py --data-dir "$DATA_DIR" --bounce-npz "$BOUNCE" \
     --tag "$TAG" --eps-s2 $EPS_S2 --eps-n $EPS_N --onset-tol $ONSET_TOL --a3-tol $A3_TOL \
     --jobs $BAND_JOBS $PIN_FLAGS
[ -f "$BAND_CUTOFFS_NPZ" ] || { echo "[ABORT] stage 1 did not produce $BAND_CUTOFFS_NPZ" | tee -a "$LOG"; exit 1; }
[ -f "$MANIFEST" ] || { echo "[ABORT] stage 1 did not produce $MANIFEST" | tee -a "$LOG"; exit 1; }
# the cutoffs of band_cutoffs, read back for the log and the resume skip-guards
# below (stages 2a-4 read them from the file themselves); the path goes in as
# sys.argv[1], never pasted into the python text, so any data-dir path is safe
N_MAX=$("$PY" -c "import sys, numpy as np; print(int(np.load(sys.argv[1])['n_max']))" "$BAND_CUTOFFS_NPZ")
S2_MAX=$("$PY" -c "import sys, numpy as np; print(float(np.load(sys.argv[1])['s2_max']))" "$BAND_CUTOFFS_NPZ")
FIT_LO=$("$PY" -c "import sys, numpy as np; print(int(np.load(sys.argv[1])['fit_lo']))" "$BAND_CUTOFFS_NPZ")
echo "  [WATCHERS] n_max=$N_MAX  s2_max=$S2_MAX  fit_lo..n_max=[$FIT_LO,$N_MAX]" | tee -a "$LOG"

stage "2a: SECTOR n=0 -> $SECTOR_N0_NPZ (s2 integral of the Born-order >= 3 trace delta_geq3 of wave 0, pole at s2 = -omega2_00 subtracted, the eigenvalue READ from eig_n0 -> Delta_n)"
# Reuse requires the same s2_max, the same floor rule (derived, or the same
# SECTOR_S2_LO) and the same convergence tolerance; a newer bounce or
# eigenvalue file invalidates it.  The band floor s2_min, which caps a derived
# floor, is not matched: the bounce and the code alone fix it.
if npz_matches "$SECTOR_N0_NPZ" s2_max $S2_MAX $SECTOR_S2_LO_MATCH sector_step_tol $SECTOR_STEP_TOL && [ ! "$BOUNCE" -nt "$SECTOR_N0_NPZ" ] && [ ! "$EIG_N0_NPZ" -nt "$SECTOR_N0_NPZ" ]; then
  echo "  [skip] $SECTOR_N0_NPZ exists at s2_max=$S2_MAX sector_s2_lo=${SECTOR_S2_LO:-derived} sector_step_tol=$SECTOR_STEP_TOL" | tee -a "$LOG"
else
  run_stage "2a sector n=0" "$PY" $COMPUTATION_DIR/gbar_sector_vgeneral_phi_v2.py --sector-n 0 --data-dir "$DATA_DIR" --bounce-npz "$BOUNCE" --tag "$TAG" \
       $SECTOR_S2_LO_FLAG --sector-step-tol $SECTOR_STEP_TOL
  [ -f "$SECTOR_N0_NPZ" ] || { echo "[ABORT] stage 2a did not write $SECTOR_N0_NPZ" | tee -a "$LOG"; exit 1; }
fi

stage "2b: SECTOR n=1 -> $SECTOR_N1_NPZ (s2 integral of the Born-order >= 3 trace delta_geq3 of wave 1, deg 4, pole at s2 = -omega2_10 subtracted, the eigenvalue READ from eig_n1 -> Delta_n)"
if npz_matches "$SECTOR_N1_NPZ" s2_max $S2_MAX $SECTOR_S2_LO_MATCH sector_step_tol $SECTOR_STEP_TOL && [ ! "$BOUNCE" -nt "$SECTOR_N1_NPZ" ] && [ ! "$EIG_N1_NPZ" -nt "$SECTOR_N1_NPZ" ]; then
  echo "  [skip] $SECTOR_N1_NPZ exists at s2_max=$S2_MAX sector_s2_lo=${SECTOR_S2_LO:-derived} sector_step_tol=$SECTOR_STEP_TOL" | tee -a "$LOG"
else
  run_stage "2b sector n=1" "$PY" $COMPUTATION_DIR/gbar_sector_vgeneral_phi_v2.py --sector-n 1 --data-dir "$DATA_DIR" --bounce-npz "$BOUNCE" --tag "$TAG" \
       $SECTOR_S2_LO_FLAG --sector-step-tol $SECTOR_STEP_TOL
  [ -f "$SECTOR_N1_NPZ" ] || { echo "[ABORT] stage 2b did not write $SECTOR_N1_NPZ" | tee -a "$LOG"; exit 1; }
fi

stage "3: BAND s2 INTEGRALS + HIGH-s2 TAIL -> $BAND_INTEGRALS_NPZ (per wave I_n_s2_max + T_n_s2_max = I_n_inf; guards MANIFEST/S2-COVERAGE/N-COVERAGE/S2-STEP/SHAPE-RATIO/REPAIR-SHARE/S2-BAACKE)"
if npz_matches "$BAND_INTEGRALS_NPZ" s2_max $S2_MAX n_max $N_MAX eps_s2 $EPS_S2 s2_step_tol $S2_STEP_TOL && [ ! "$BOUNCE" -nt "$BAND_INTEGRALS_NPZ" ] && [ -z "$(find "$DATA_DIR" -maxdepth 1 -name "residual_band_${TAG}_s2*.npz" -newer "$BAND_INTEGRALS_NPZ" -print -quit)" ]; then
  echo "  [skip] $BAND_INTEGRALS_NPZ exists at (n_max=$N_MAX, s2_max=$S2_MAX)" | tee -a "$LOG"
else
  run_stage "3 band integrals" "$PY" $COMPUTATION_DIR/band_integrals_vgeneral_phi_v2.py --data-dir "$DATA_DIR" --bounce-npz "$BOUNCE" --tag "$TAG" \
       --s2-step-tol $S2_STEP_TOL
  [ -f "$BAND_INTEGRALS_NPZ" ] || { echo "[ABORT] stage 3 did not write $BAND_INTEGRALS_NPZ" | tee -a "$LOG"; exit 1; }
fi

stage "4: HIGH-N TAIL -> $TAIL_HIGH_N_NPZ (ONE odd-family fit a/nu^3+b/nu^5+c/nu^7 of the s2-completed I_n(inf) on [$FIT_LO,$N_MAX] + the Hurwitz-zeta sum Zcal over n > $N_MAX; guards N-WINDOW/N-ASYMPTOTE/N-TAIL-BOUND)"
if npz_matches "$TAIL_HIGH_N_NPZ" fit_lo $FIT_LO n_max $N_MAX a3_tol $A3_TOL eps_n $EPS_N && [ ! "$BAND_INTEGRALS_NPZ" -nt "$TAIL_HIGH_N_NPZ" ]; then
  echo "  [skip] $TAIL_HIGH_N_NPZ exists at fit_lo..n_max=[$FIT_LO,$N_MAX] and is newer than stage 3" | tee -a "$LOG"
else
  run_stage "4 high-n tail" "$PY" $COMPUTATION_DIR/tail_high_n_zeta_vgeneral_phi_v2.py --data-dir "$DATA_DIR" --bounce-npz "$BOUNCE" --tag "$TAG"
  [ -f "$TAIL_HIGH_N_NPZ" ] || { echo "[ABORT] stage 4 did not write $TAIL_HIGH_N_NPZ" | tee -a "$LOG"; exit 1; }
fi

stage "5: TADPOLE ADD-BACK -> $ADDBACK_TADPOLE_NPZ (A1_fin = sum_i M_i tau_fin_fv(m_i^2), the traced Born-order-1 add-back in closed form, at mu = sum_i m_i)"
if [ -f "$ADDBACK_TADPOLE_NPZ" ] && [ ! "$BOUNCE" -nt "$ADDBACK_TADPOLE_NPZ" ]; then
  echo "  [skip] $ADDBACK_TADPOLE_NPZ exists" | tee -a "$LOG"
else
  run_stage "5 tadpole add-back" "$PY" $COMPUTATION_DIR/addback_tadpole_vgeneral_phi_v2.py --data-dir "$DATA_DIR" --bounce-npz "$BOUNCE" --tag "$TAG"
  [ -f "$ADDBACK_TADPOLE_NPZ" ] || { echo "[ABORT] stage 5 did not write $ADDBACK_TADPOLE_NPZ" | tee -a "$LOG"; exit 1; }
fi

stage "6: FISH ADD-BACK -> $ADDBACK_FISH_NPZ (A2_fin, the MS-bar finite part of Born order 2, which enters lnD_ren as -A2_fin/2; two converged ladders, guard FISH-CONVERGED)"
if [ -f "$ADDBACK_FISH_NPZ" ] && [ ! "$BOUNCE" -nt "$ADDBACK_FISH_NPZ" ]; then
  echo "  [skip] $ADDBACK_FISH_NPZ exists" | tee -a "$LOG"
else
  run_stage "6 fish add-back" "$PY" $COMPUTATION_DIR/addback_fish_vgeneral_phi_v2.py --data-dir "$DATA_DIR" --bounce-npz "$BOUNCE" --tag "$TAG"
  [ -f "$ADDBACK_FISH_NPZ" ] || { echo "[ABORT] stage 6 did not write $ADDBACK_FISH_NPZ" | tee -a "$LOG"; exit 1; }
fi

stage "7: ASSEMBLE lnD_ren -> $LND_REN_NPZ (the seven terms (T1)-(T7); guards SECTOR-CUTOFF/FICT-BLOCK/LND-FINITE)"
# Skip ONLY if the assembly exists AND is newer than every input npz (a stale
# assembly must not survive an input recompute; the eig npz feed the sector
# npz, so they are inputs too).
NEED_ASSEMBLY=0
if [ ! -f "$LND_REN_NPZ" ]; then
  NEED_ASSEMBLY=1
else
  for f in "$BOUNCE" "$EIG_N0_NPZ" "$EIG_N1_NPZ" "$SECTOR_N0_NPZ" "$SECTOR_N1_NPZ" "$BAND_INTEGRALS_NPZ" "$TAIL_HIGH_N_NPZ" "$ADDBACK_TADPOLE_NPZ" "$ADDBACK_FISH_NPZ"; do
    if [ "$f" -nt "$LND_REN_NPZ" ]; then NEED_ASSEMBLY=1; fi
  done
fi
if [ "$NEED_ASSEMBLY" = 0 ]; then
  echo "  [skip] $LND_REN_NPZ exists and is newer than all inputs" | tee -a "$LOG"
else
  run_stage "7 assemble" "$PY" $COMPUTATION_DIR/assemble_lnD_vgeneral_phi_v2.py --data-dir "$DATA_DIR" --bounce-npz "$BOUNCE" --tag "$TAG"
  [ -f "$LND_REN_NPZ" ] || { echo "[ABORT] stage 7 did not write $LND_REN_NPZ" | tee -a "$LOG"; exit 1; }
fi

echo "[$(ts)] vgeneral_phi_v2 PRODUCTION DONE -> $LND_REN_NPZ (watchers found n_max=$N_MAX s2_max=$S2_MAX fit_lo=$FIT_LO)" | tee -a "$LOG"
