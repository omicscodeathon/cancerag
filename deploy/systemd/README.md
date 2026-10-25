# Run-host systemd units

These are the units on `cancerag-run` (GCE, `c2d-standard-8`, `europe-west4-a`).
They existed only on that machine until 2026-10-25, which is the point of this
directory: a unit that is not in the repository is not reviewed, is not
reproducible, and — as below — can be hard even to find.

| file | what it does |
| --- | --- |
| `cancerag-zenml.service` | local ZenML server on :8237, backing the pipeline's DAG and dashboard |
| `cancerag-run.service` | one full `python -m cancerag.pipeline` run |
| `cancerag-publish.service` / `.timer` | every 60 s, mirror `status.json` and a 200 kB tail of `pipeline.log` to the run bucket |
| `cancerag-publish` | the script the publish service runs, installed to `/usr/local/bin` |

Install with `sudo deploy/systemd/install.sh`. Nothing is enabled unless you
pass `--enable`.

## Why `cancerag-run` has a start limit

On 2026-10-25 `systemctl show cancerag-run.service -p NRestarts` read **2486**.

The unit had `Restart=on-failure` and `RestartSec=60` in an unversioned
drop-in, with no `StartLimitBurst`. The pipeline was failing at its last step
(`figures`, HAL-4148): it ran everything, failed, exited non-zero, and systemd
restarted it a minute later. Resume keys on artefact presence, so the retry
skipped every completed stage and arrived back at the same failing step in
seconds. That cycle ran about every 71 s for several days on a billed machine.

Two things made it expensive beyond the compute:

- **The run looked healthy.** `systemctl status` said `activating`, the log kept
  growing, and new artefacts kept appearing. Nothing said "this has failed 2486
  times".
- **It was hard to kill.** Killing the pipeline returned it a minute later,
  reparented to PID 1 with its launcher already gone, which looks exactly like
  an unknown supervisor. Finding it needed a `systemctl list-timers` sweep.

`StartLimitIntervalSec=3600` + `StartLimitBurst=3` keeps a retry for genuinely
transient faults and stops a deterministic one after two, leaving the unit in
`failed` where it is visible.

## Operating notes

- `pipeline.log` is append-only across every run and reached 101 MB. The
  publisher uploads only a tail; rotate the file if it grows further.
- The pipeline resumes on **artefact presence, not freshness**, so a stale
  output is treated as done. Use `--force` after changing anything upstream.
  This is HAL-4141.
- To run one stage: `--only <step>`; to run from one onward: `--from <step>`.
  `--list` prints the chain and what is already built.
