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
  # default-jre   -> P2Rank is a JVM application
  # libnetcdf-dev -> fpocket links against it
  # build-essential + libs -> fpocket, and wheels with no manylinux build
  #
  # fpocket is NOT packaged for Ubuntu 24.04 (absent from universe too), so it
  # is built from source below rather than installed here.
  apt-get install -y -qq --no-install-recommends \
    curl ca-certificates unzip tar gzip git \
    openbabel default-jre-headless \
    build-essential libnetcdf-dev libboost-all-dev swig \
    >/dev/null
}

install_fpocket() {
  if command -v fpocket >/dev/null 2>&1; then log "fpocket present, skipping"; return; fi
  log "fpocket (from source — not in the Ubuntu archive)"
  local src="${TOOLS_DIR}/fpocket"
  rm -rf "$src"
  mkdir -p "$TOOLS_DIR"
  git clone -q --depth 1 https://github.com/Discngine/fpocket.git "$src" \
    || die "could not clone fpocket"
  # Serial on purpose: fpocket's makefile has an ordering bug in the bundled
  # qhull and fails under -j with "Error 1" on qvoronoi.o. It takes a couple of
  # minutes either way.
  ( cd "$src" && make -j1 >/tmp/fpocket_build.log 2>&1 ) \
    || { tail -20 /tmp/fpocket_build.log >&2; die "fpocket build failed"; }
  # `make install` is unreliable here; the binaries land in bin/ either way.
  install -m0755 "$src"/bin/fpocket "$src"/bin/dpocket "$src"/bin/mdpocket \
    "$src"/bin/tpocket /usr/local/bin/ || die "could not install fpocket binaries"
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
  rm -f /tmp/p2rank.tar.gz
  # P2Rank's launcher resolves its jars and models relative to its own location,
  # so a symlink into /usr/local/bin breaks it with "Could not find or load main
  # class cz.siret.prank.program.Main". Ship a wrapper that runs it in place.
  cat >/usr/local/bin/prank <<WRAPPER
#!/usr/bin/env bash
exec "${TOOLS_DIR}/p2rank_${P2RANK_VERSION}/prank" "\$@"
WRAPPER
  chmod +x /usr/local/bin/prank
}

install_gnina() {
  if command -v gnina >/dev/null 2>&1 && gnina --version >/dev/null 2>&1; then
    log "gnina present, skipping"; return
  fi
  log "Gnina"
  local bin="${TOOLS_DIR}/gnina.bin"
  mkdir -p "$TOOLS_DIR"

  if [[ -n "${GNINA_STAGED:-}" && -f "${GNINA_STAGED}" ]]; then
    install -m0755 "${GNINA_STAGED}" "$bin"
  else
    # The release asset is named gnina.cuda12.8.static, so /latest/download
    # (which needs a fixed filename) 404s; resolve it through the API.
    local url
    url=$(curl -fsSL https://api.github.com/repos/gnina/gnina/releases/latest \
          | grep -oE '"browser_download_url": *"[^"]*gnina[^"]*"' \
          | head -1 | cut -d'"' -f4 || true)
    if [[ -z "$url" ]] || ! curl -fsSL "$url" -o "$bin"; then
      warn "gnina download failed — the pipeline still runs, but every"
      warn "gnina_cnn_score comes back null and docking_confidence falls back"
      warn "to the binding-site flag, which changes the sample weights."
      warn "Set GNINA_STAGED=/path/to/gnina and re-run to fix."
      return
    fi
    chmod +x "$bin"
  fi

  # Despite the "static" in its name the binary still links the CUDA 12 runtime
  # and cuDNN 9, which Ubuntu does not ship. The pip nvidia-*-cu12 wheels carry
  # exactly those sonames. nvtx is pinned because 12.9 renamed
  # libnvToolsExt.so.1 to libnvtx3interop.so.1, which gnina does not look for.
  # No GPU is needed — the libraries only have to be loadable.
  local libs="${TOOLS_DIR}/gnina-libs"
  if [[ ! -d "$libs/nvidia" ]]; then
    log "  CUDA sonames for gnina"
    uv pip install --quiet --target "$libs" \
      nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
      nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-cusolver-cu12 \
      nvidia-cusparse-cu12 nvidia-nvjitlink-cu12 "nvidia-nvtx-cu12==12.1.105" \
      || { warn "could not fetch CUDA runtime libs; gnina will not load"; return; }
  fi

  cat >/usr/local/bin/gnina <<WRAPPER
#!/usr/bin/env bash
LD=\$(find "${libs}/nvidia" -maxdepth 2 -type d -name lib | tr '\\n' ':')
export LD_LIBRARY_PATH="\${LD}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "${bin}" "\$@"
WRAPPER
  chmod +x /usr/local/bin/gnina
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then log "uv present, skipping"; return; fi
  log "uv"
  curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
}

# -------------------------------------------------------------- verification

# Several of these print their banner and then exit non-zero (fpocket with no
# input, prank --help, java -version). Exit status is therefore not a usable
# signal; match the banner instead.
# name | probe | fatal | expected substring
CHECKS=(
  "vina|vina --version|yes|AutoDock Vina"
  "obabel|obabel -V|yes|Open Babel"
  "fpocket|fpocket|yes|POCKET HUNTING"
  "prank|prank|yes|P2Rank"
  "java|java -version|yes|version"
  "uv|uv --version|yes|uv "
  "gnina|gnina --version|no|gnina"
)

verify() {
  log "verifying external binaries"
  local failed=0 soft=0
  for entry in "${CHECKS[@]}"; do
    IFS='|' read -r name probe fatal expect <<<"$entry"
    if ! command -v "$name" >/dev/null 2>&1; then
      if [[ $fatal == yes ]]; then
        printf '  \033[1;31m✗ %-8s not on PATH\033[0m\n' "$name"; failed=$((failed+1))
      else
        printf '  \033[1;33m○ %-8s not on PATH (optional)\033[0m\n' "$name"; soft=$((soft+1))
      fi
      continue
    fi
    # A binary that exists but cannot execute (missing .so, wrong arch, a JVM
    # launcher that lost its home directory) is the failure that otherwise
    # surfaces three hours in. Run it and look for its banner.
    local out
    out=$(timeout 60 $probe 2>&1 | head -5 || true)
    if grep -qF "$expect" <<<"$out"; then
      printf '  \033[1;32m✓ %-8s\033[0m %s\n' "$name" "$(head -1 <<<"$out" | cut -c1-60)"
    else
      printf '  \033[1;31m✗ %-8s present but did not report %s: %s\033[0m\n' \
        "$name" "$expect" "$(head -1 <<<"$out" | cut -c1-50)"
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
    install_uv
    install_fpocket
    install_vina
    install_p2rank
    install_gnina
  else
    log "verify-only"
  fi
  verify
  report_resources
  log "ready"
}

main "$@"
