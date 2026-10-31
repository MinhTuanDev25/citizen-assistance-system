#!/usr/bin/env bash
# /workspace is a plain directory here (no volume mounted): check, preserve and apply must
# FAIL volume-mount and write nothing under /workspace.
set -uo pipefail
pass=0 failn=0
ok()  { echo "  ok   $1"; pass=$((pass + 1)); }
no()  { echo "  FAIL $1"; failn=$((failn + 1)); }
mkdir -p /workspace /root/.jupyter && echo cfg > /root/.jupyter/c.py
before="$(find /workspace | sort)"
bash /staging/pod_stage.sh check > /tmp/c 2>&1; rc=$?
[[ $rc == 1 ]] && grep -q '^FAIL  volume-mount .*not a mount point' /tmp/c && ok "T11 check: FAIL volume-mount (rc $rc)" || no "T11 check (rc $rc)"
bash /staging/pod_stage.sh preserve /root/.jupyter > /tmp/p 2>&1; rc=$?
[[ $rc == 1 ]] && grep -q '^FAIL  volume-mount' /tmp/p && grep -q 'nothing written' /tmp/p && ok "T12 preserve: refused (rc $rc)" || no "T12 preserve (rc $rc): $(tail -2 /tmp/p)"
[[ ! -e /workspace/cas ]] && ok "T12 preserve wrote no backup (/workspace/cas absent)" || no "T12 preserve created /workspace/cas"
bash /staging/pod_stage.sh preserve > /tmp/p2 2>&1; rc=$?
[[ $rc == 1 ]] && ok "T12b preserve without paths: refused (rc $rc)" || no "T12b preserve without paths (rc $rc)"
bash /staging/pod_stage.sh apply > /tmp/a 2>&1; rc=$?
[[ $rc == 1 ]] && grep -q "STOPPED at step 'volume mount" /tmp/a && ok "T13 apply: STOPPED at volume mount (rc $rc)" || no "T13 apply (rc $rc): $(grep -E 'STOPPED|FAIL' /tmp/a | head -3)"
[[ "$(find /workspace | sort)" == "$before" ]] && ok "T11-13 /workspace unchanged" || no "T11-13 /workspace changed: $(find /workspace)"
echo "NOMOUNT: $pass passed, $failn failed"
exit $(( failn > 0 ))
