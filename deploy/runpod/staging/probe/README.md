# Filesystem probe v2

`probe_v2.py` is a measurement tool, not a gate. It runs on the existing pod, still on the old PyTorch image, and tells us what the network volume at `/workspace` actually supports, one property at a time. It changes nothing in the deployment, and it does not use or need S3 credentials.

## What it checks

Every check prints `PASS`, `FAIL`, `SKIP` or `INFO` with its evidence. A failing check does not stop the others.

| Group | Checks |
|---|---|
| Environment | Is `/workspace` a mount point, and which filesystem and source? Size of the volume backend and of the container disk. `CapEff` capabilities. Presence of `/dev/loop-control`, `/dev/loop0` and `/dev/fuse` (no mount is ever attempted) |
| Modes and owners | Does `chmod 0700` on a directory and `chmod 0600` on a file take effect? Does `chown` by root work? Which owner does root see on a file created by uid 10010? Can uid 10010 read a root file at mode 0600 (isolation between service uids)? |
| mkdir | Does a second `mkdir` raise `EEXIST`? 100 rounds of two processes racing for one name: exactly one must win each round |
| rename | `os.replace` over an existing file. 300 replaces while another process reads 3000 times: the reader must never see a missing or partial file. Rename of a directory |
| Locks | `flock` and `fcntl` exclusive locks: a second process must be refused while the lock is held, and get it after release |
| I/O | O_DIRECT aligned write and read; `MAP_SHARED` mmap write seen by `read()` in another process; `fsync` on a directory; `link()` |
| fsync | 200 × (8 KB write + fsync): p50/p95/p99/max. 8 × (16 MB write + fsync), the size of one WAL segment. 4 processes × 50 × (8 KB + fsync) at the same time |
| Throughput | 256 MB sequential write + fsync, then a read |
| Cleanup | The probe directory is gone, and every other entry under the root is unchanged |

## Safety

- It writes only random test data, inside a new `/workspace/cas/.probe-<utc>-<hex>` created with `mkdir`, so it never reuses a directory.
- It refuses a root that is a symlink or resolves elsewhere. It never follows symlinks.
- It does not touch `preserve/`, `models/` or anything outside the probe directory. The research tree is outside `/workspace/cas` and is never opened.
- **Byte budget, two layers:**
  - Every write is planned against `--max-bytes`, by default and at most 600 MB in total; this version writes about 392 MB. A write that would go over the budget is not made, and its check is a `SKIP`.
  - The probe directory is also **measured** (apparent size and allocated blocks) after every benchmark and inside the large writers. If it uses more than `--footprint-cap-mb` (default and maximum 300 MB; in practice 256 MB), every remaining check stops with `FAIL budget.hard-stop`, and the cleanup still runs.
- **Child processes:**
  - Every forked check process is tracked. Each one always ends in `os._exit`, so an error in a child can never run the parent's code.
  - On Ctrl-C or SIGTERM, all of them are killed and reaped **before** the probe directory is removed.
  - After each check group, a leftover child is a `FAIL safety.children`.
- **On exit,** it verifies three things:
  - the probe directory is gone;
  - no check process is left (`cleanup.children`);
  - `/workspace/cas` is otherwise unchanged.
- It needs Linux and Python ≥ 3.9; otherwise it refuses to start.
- Exit codes:
  - `0`: probe and cleanup completed, whatever the PASS/FAIL results;
  - `1`: cleanup could not be verified;
  - `2`: refused to start, nothing written;
  - `130`: interrupted, children reaped, probe directory removed.

## What it cannot show

- **That a successful `fsync` survives a host or power failure.** It measures that the call returns, and how fast.
- **Locking or atomicity between two pods on two hosts.** Every concurrency check runs on one host and one kernel.
- **Anything about the S3 API.** That is a separate step and is not approved.

None of its results decides between the storage options on its own.

## Run on the pod (as root, old image)

```bash
cd /tmp
python3 /tmp/cas-src/deploy/runpod/staging/probe/probe_v2.py --json 2>&1 | tee /tmp/probe-v2.txt; echo "exit=${PIPESTATUS[0]}"
```

Stop RQ jobs that write to `/workspace` first. They do not affect safety, but they add noise to the latency numbers.

## Tests

`tests/run_probe_tests.sh` runs `tests/probe_tests.sh` in throwaway containers, once on an ext4 volume and once on bindfs. bindfs is mounted with `--perms=a+rwx --chmod-ignore --chown-deny`, which mimics the pod's 0777 and EPERM. Every expectation is asserted:

- **Full run.** The expected FAIL set on each filesystem:
  - ext4: only `env.loop-mount-possible`;
  - bindfs: modes, `chown`, isolation, O_DIRECT, loop, plus `rename.atomic-under-reads`, which depends on timing there.
  - The cleanup lines PASS, and the measured footprint is reported.
- **Interrupts.** SIGTERM and SIGINT during the concurrent fsync, and SIGTERM during rename under reads:
  - exit code 130;
  - the children were running when the signal came, and they were killed and reaped (0 left);
  - no probe process and no `.probe-*` remain;
  - the rest of the root is unchanged.
- **Budget.**
  - A planned write over the cap is SKIPped and writes 0 bytes.
  - Unexpected data planted in the probe directory during a run triggers the measured hard stop: no check runs after it, and the cleanup removes it.
  - `--max-bytes 10 MB` keeps the total at or under 10 MB.
- **A failing child never runs the parent's code**, whether it raises ZeroDivisionError, raises ENOSPC, or returns an int.
- **Long concurrent results** (4 × 3000 values) parse correctly, with one pipe per worker.
- **Refused arguments** write nothing.

The last run gave `PROBE-TESTS ext4: 44 passed, 0 failed` and `PROBE-TESTS bindfs: 44 passed, 0 failed`. These local results do not predict MooseFS; only the run on the pod does.

The options `--only`, `--rename-writes`, `--fsync-n`, `--concurrent-n`, `--footprint-cap-mb` and `--max-bytes` exist for these tests. On the pod, run with the defaults.
