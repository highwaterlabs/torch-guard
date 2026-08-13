# design/

Working notes for building torch-guard. Not user documentation — [README.md](../README.md)
is the user-facing doc, and nothing in here ships in the wheel.

**This directory is public**, and deliberately so: the RFCs and spike write-ups are the
evidence behind the numbers the tool prints, which is most of why anyone should trust an
OOM prediction. Commercial design notes live in the private `torch-guard-cloud` repo
instead, in the private `torch-guard-cloud` repo — including the RFC that defines the
free/paid boundary, which is itself on the private side of it.

| Folder | What goes in it | Lifetime |
|---|---|---|
| [TODO.md](TODO.md) | Actionable backlog, grouped by phase. Things we have decided to do. | Items get deleted when done |
| [IDEAS.md](IDEAS.md) | Unfiltered parking lot. No commitment implied. | Grows freely; promote to TODO or RFC when real |
| [rfcs/](rfcs/) | Designs big enough to need agreement *before* code exists | Permanent record, superseded not deleted |
| — [0001](rfcs/0001-vram-estimator.md) | Pre-flight VRAM estimation | Implemented |
| [spikes/](spikes/) | Time-boxed experiments answering one uncertain question | Permanent record of what we learned |

Measurement scripts live in [`tests/calibration/`](../tests/calibration/), not here — they
are public tooling, since the calibration numbers are only credible if anyone can
reproduce them.

## When to write which

**RFC** — the change touches architecture, adds a dependency, changes a public
interface, or would be expensive to undo. Write it, get agreement, then build.
An RFC is cheap; rewriting a subsystem is not.

**Spike** — we do not know if something is technically possible, and the answer changes
the design. Time-box it, write down the answer, throw the code away. A spike that turns
into production code is a spike that was not a spike.

**TODO** — we already know what to do and how.

**IDEA** — worth remembering, not worth deciding on yet.

## Conventions

- RFCs and spikes are numbered sequentially and never renumbered: `0001-short-slug.md`.
- Every RFC carries a `Status:` line — `Draft` · `Accepted` · `Implemented` · `Superseded by NNNN`.
- Update the status when reality changes. A stale `Accepted` on something we abandoned is
  worse than no document.
- Record decisions *with their reasoning*. Six months from now the reasoning is the only
  part that still has value.
