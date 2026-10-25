#!/usr/bin/env bash
# Install the CancerAg systemd units on the run host.
#
#   sudo deploy/systemd/install.sh            # install, do not start anything
#   sudo deploy/systemd/install.sh --enable   # install and enable the timer + run
#
# Nothing is enabled by default. A batch pipeline that starts itself on boot is
# how a failing run becomes an invisible one: see the 2026-10-25 incident in
# cancerag-run.service, where a unit nobody was watching retried the same
# failure 2486 times.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR=/etc/systemd/system
BIN_DIR=/usr/local/bin

if [ "$(id -u)" -ne 0 ]; then
  echo "run with sudo" >&2
  exit 1
fi

for unit in cancerag-zenml.service cancerag-run.service \
            cancerag-publish.service cancerag-publish.timer; do
  install -m 0644 "$HERE/$unit" "$UNIT_DIR/$unit"
  echo "installed $UNIT_DIR/$unit"
done

install -m 0755 "$HERE/cancerag-publish" "$BIN_DIR/cancerag-publish"
echo "installed $BIN_DIR/cancerag-publish"

# A drop-in from an earlier hand-rolled install carried Restart=on-failure with
# no start limit. It would silently override what is now in the unit file, so
# it goes.
if [ -d "$UNIT_DIR/cancerag-run.service.d" ]; then
  rm -rf "$UNIT_DIR/cancerag-run.service.d"
  echo "removed the unversioned cancerag-run.service.d drop-in"
fi

systemctl daemon-reload
echo "daemon-reloaded"

if [ "${1:-}" = "--enable" ]; then
  systemctl enable --now cancerag-zenml.service
  systemctl enable --now cancerag-publish.timer
  systemctl enable cancerag-run.service
  echo "enabled zenml + publish timer; cancerag-run enabled but not started"
  echo "start a run with: sudo systemctl start cancerag-run"
else
  echo
  echo "Nothing enabled. To enable:  sudo $0 --enable"
fi

echo
echo "State:"
systemctl is-enabled cancerag-run.service 2>/dev/null || true
systemctl is-active  cancerag-run.service 2>/dev/null || true
