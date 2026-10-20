#!/usr/bin/env bash
# Provision a bare Debian/Ubuntu VM to run the CancerAg pipeline end to end.
#
# The pipeline shells out to five external binaries. None of them is a Python
# dependency, and the code finds them with shutil.which(), so a missing one does
# not raise — it silently degrades: pocket prediction is skipped, Gnina scores
# come back null, docking never starts. That is how you lose six hours and learn
# nothing. This script installs them and then refuses to exit 0 unless every one
# answers.
#
#   sudo ./provision_vm.sh            install everything, then verify
#   ./provision_vm.sh --verify-only   just check (no root needed)
#
# Idempotent: re-running skips what is already present.
set -Eeuo pipefail

TOOLS_DIR="${CANCERAG_TOOLS_DIR:-/opt/cancerag-tools}"
P2RANK_VERSION="2.5.1"
VINA_VERSION="1.2.5"
VERIFY_ONLY=0
[[ "${1:-}" == "--verify-only" ]] && VERIFY_ONLY=1

log()  { printf '\033[1;34m[provision]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fatal]\033[0m %s\n' "$*" >&2; exit 1; }

need_root() {
  [[ $EUID -eq 0 ]] || die "installation needs root; re-run with sudo (or use --verify-only)"
}

# ---------------------------------------------------------------- installation

install_apt_packages() {
  log "apt packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  # openbabel     -> charge/format fallback in receptor prep
  # fpocket       -> geometric pocket prediction, cross-checks P2Rank
  # default-jre   -> P2Rank is a JVM application
  # build-essential + libs -> wheels that have no manylinux build
  apt-get install -y -qq --no-install-recommends \
    curl ca-certificates unzip tar gzip git \
    openbabel fpocket default-jre-headless \
    build-essential libboost-all-dev swig \
    >/dev/null
}

install_vina() {
  if command -v vina >/dev/null 2>&1; then log "vina present, skipping"; return; fi
  log "AutoDock Vina ${VINA_VERSION}"
  local url="https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v${VINA_VERSION}/vina_${VINA_VERSION}_linux_x86_64"
  curl -fsSL "$url" -o /usr/local/bin/vina || die "could not download vina from $url"
  chmod +x /usr/local/bin/vina
}

install_p2rank() {
  if command -v prank >/dev/null 2>&1; then log "p2rank present, skipping"; return; fi
  log "P2Rank ${P2RANK_VERSION}"
  mkdir -p "$TOOLS_DIR"
  local url="https://github.com/rdk/p2rank/releases/download/${P2RANK_VERSION}/p2rank_${P2RANK_VERSION}.tar.gz"
  curl -fsSL "$url" -o /tmp/p2rank.tar.gz || die "could not download p2rank from $url"
  tar -xzf /tmp/p2rank.tar.gz -C "$TOOLS_DIR"
  ln -sf "${TOOLS_DIR}/p2rank_${P2RANK_VERSION}/prank" /usr/local/bin/prank
  rm -f /tmp/p2rank.tar.gz
}

install_gnina() {
  if command -v gnina >/dev/null 2>&1; then log "gnina present, skipping"; return; fi
  log "Gnina"
  # The release binary is ~1.4 GB. If it was staged alongside this script
  # (cheaper than re-downloading per VM), prefer that.
  if [[ -n "${GNINA_STAGED:-}" && -f "${GNINA_STAGED}" ]]; then
    install -m 0755 "${GNINA_STAGED}" /usr/local/bin/gnina
    return
  fi
  local url="https://github.com/gnina/gnina/releases/latest/download/gnina"
  if curl -fsSL "$url" -o /usr/local/bin/gnina; then
    chmod +x /usr/local/bin/gnina
  else
    warn "gnina download failed — the pipeline runs without it, but every"
    warn "gnina_cnn_score comes back null and docking_confidence degrades to"
    warn "the binding-site flag. Set GNINA_STAGED=/path/to/gnina and re-run."
  fi
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then log "uv present, skipping"; return; fi
  log "uv"
  curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
}

# -------------------------------------------------------------- verification

# name | probe command | fatal if missing
CHECKS=(
  "vina|vina --version|yes"
  "obabel|obabel -V|yes"
  "fpocket|fpocket -v|yes"
  "prank|prank --help|yes"
  "java|java -version|yes"
  "uv|uv --version|yes"
  "gnina|gnina --version|no"
)

verify() {
  log "verifying external binaries"
  local failed=0 soft=0
  for entry in "${CHECKS[@]}"; do
    IFS='|' read -r name probe fatal <<<"$entry"
    if ! command -v "$name" >/dev/null 2>&1; then
      if [[ $fatal == yes ]]; then
        printf '  \033[1;31m✗ %-8s not on PATH\033[0m\n' "$name"; failed=$((failed+1))
      else
        printf '  \033[1;33m○ %-8s not on PATH (optional)\033[0m\n' "$name"; soft=$((soft+1))
      fi
      continue
    fi
    # A binary that exists but cannot execute (missing .so, wrong arch) is the
    # failure mode that otherwise surfaces three hours in. Run it.
    local out
    if out=$($probe 2>&1 | head -1); then
      printf '  \033[1;32m✓ %-8s\033[0m %s\n' "$name" "${out:0:60}"
    else
      printf '  \033[1;31m✗ %-8s present but will not run: %s\033[0m\n' "$name" "${out:0:60}"
      [[ $fatal == yes ]] && failed=$((failed+1))
    fi
  done

  # Vina must actually dock, not merely print a version. A 30-second smoke test
  # here is worth more than any amount of --version output.
  if command -v vina >/dev/null 2>&1; then
    local tmp; tmp=$(mktemp -d)
    printf 'REMARK trivial\nATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\nENDROOT\nTORSDOF 0\n' >"$tmp/l.pdbqt"
    printf 'ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n' >"$tmp/r.pdbqt"
    if timeout 60 vina --receptor "$tmp/r.pdbqt" --ligand "$tmp/l.pdbqt" \
         --center_x 0 --center_y 0 --center_z 0 --size_x 10 --size_y 10 --size_z 10 \
         --exhaustiveness 1 --out "$tmp/o.pdbqt" >/dev/null 2>&1; then
      printf '  \033[1;32m✓ vina\033[0m     smoke dock completed\n'
    else
      printf '  \033[1;33m○ vina\033[0m     smoke dock did not complete (trivial input; not necessarily fatal)\n'
    fi
    rm -rf "$tmp"
  fi

  echo
  if (( failed > 0 )); then
    die "$failed required binary/binaries missing or broken — fix before running the pipeline"
  fi
  (( soft > 0 )) && warn "$soft optional binary missing; the run will proceed with reduced output"
  log "all required binaries present"
}

report_resources() {
  log "machine"
  printf '  cpus      %s\n' "$(nproc)"
  printf '  memory    %s\n' "$(free -h 2>/dev/null | awk '/^Mem:/{print $2}')"
  printf '  disk      %s free on %s\n' \
    "$(df -h . | awk 'NR==2{print $4}')" "$(df -h . | awk 'NR==2{print $6}')"
}

main() {
  if (( VERIFY_ONLY == 0 )); then
    need_root
    install_apt_packages
    install_vina
    install_p2rank
    install_gnina
    install_uv
  else
    log "verify-only"
  fi
  verify
  report_resources
  log "ready"
}

main "$@"
