"""Measure what a retained autograd graph actually holds, with and without a backward.

    python tests/calibration/measure_retention.py            # table
    python tests/calibration/measure_retention.py --json     # machine-readable

Why this exists
---------------
TG001 used to report every graph-attached store as ``severity=error`` /
``category=CRITICAL_OOM``, with the message "every iteration's graph is retained in VRAM ...
until the run OOMs". Triaging real training repos showed that claim holds for only some of
what it flagged, and the difference is whether the stored tensor has been backwarded.

``backward()`` releases each node's saved tensors as it traverses. So::

    loss.backward()
    losses.append(loss)     # activations already freed; graph *nodes* retained

is a much smaller leak than::

    losses.append(loss)     # nothing backwards this
    # ... no backward anywhere

which retains the activations as well — the OOM the rule describes. Same one-line fix,
different order of magnitude, and different memory: nodes are host-side bookkeeping, saved
activations live wherever the tensors do, which on a training run is VRAM.

The rule now splits on it (error vs warning), so the split needs a measurement rather than
an argument.

Method
------
Walks the autograd graph still reachable from the kept tensors and sums the ``_saved_*``
tensors each node holds. That is a **direct, deterministic** count of what retention costs:
the same input gives the same answer every run, and it measures the activations themselves
rather than a proxy.

RSS was the obvious instrument and is the wrong one — ``ru_maxrss`` is a high-water mark
perturbed by allocator arenas and GC timing, and three runs of the same code spread over
5.8-20.9 KiB/iteration. Anything quoted from a number that unstable would be noise. Node
counts are reported instead of node bytes, since the nodes are C++ objects with no honest
Python-side size.

CPU-only and takes a few seconds, so unlike the other harnesses here this one needs no GPU.
"""

from __future__ import annotations

import argparse
import json

ITERATIONS = 8
DEPTH, WIDTH, BATCH = 12, 256, 16


def _graph_footprint(tensors):
    """Saved-tensor bytes and node count reachable from ``tensors``.

    Every autograd node exposes what it stashed for backward as ``_saved_<name>``
    attributes. Reading one after the buffers are freed raises, which is itself the
    signal we are after, so those are counted as zero.
    """
    import torch

    seen_nodes, seen_storages = set(), set()
    total = 0
    stack = [t.grad_fn for t in tensors if getattr(t, "grad_fn", None) is not None]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node))
        for attribute in dir(node):
            if not attribute.startswith("_saved_"):
                continue
            try:
                value = getattr(node, attribute)
            except RuntimeError:
                continue  # "saved tensors have already been freed" -- exactly the point
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                key = value.untyped_storage().data_ptr()
                if key not in seen_storages:
                    seen_storages.add(key)
                    total += value.numel() * value.element_size()
        stack.extend(n for n, _ in (node.next_functions or ()))
    return total, len(seen_nodes)


def measure(mode: str, iterations: int = ITERATIONS) -> dict:
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Sequential(
        *[nn.Linear(WIDTH, WIDTH) for _ in range(DEPTH)], nn.Linear(WIDTH, 1)
    )
    criterion = nn.MSELoss()

    kept = []
    for _ in range(iterations):
        loss = criterion(model(torch.randn(BATCH, WIDTH)), torch.randn(BATCH, 1))
        if mode != "never_backwarded":
            loss.backward()
        kept.append(loss.item() if mode == "detached" else loss)

    saved_bytes, nodes = _graph_footprint(kept)
    return {
        "mode": mode,
        "iterations": iterations,
        "retained_activation_bytes": saved_bytes,
        "retained_graph_nodes": nodes,
        "activation_bytes_per_iteration": saved_bytes / iterations,
        "nodes_per_iteration": nodes / iterations,
    }


def second_backward_is_refused() -> str:
    """Independent confirmation that the buffers are gone, from torch's own error."""
    import torch
    import torch.nn as nn

    model = nn.Linear(WIDTH, 1)
    loss = nn.MSELoss()(model(torch.randn(BATCH, WIDTH)), torch.randn(BATCH, 1))
    loss.backward()
    try:
        loss.backward()
    except RuntimeError as error:
        return str(error).split(".")[0]
    return "second backward succeeded -- buffers were retained"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = [measure(m) for m in ("detached", "backwarded", "never_backwarded")]
    if args.json:
        print(json.dumps(
            {"results": results, "second_backward": second_backward_is_refused()}, indent=2
        ))
        return 0

    print(f"{DEPTH + 1}x{WIDTH} MLP, batch {BATCH}, {ITERATIONS} iterations, one loss kept per step\n")
    print(f"{'mode':18}{'activations held':>19}{'per iteration':>16}{'nodes':>8}")
    for row in results:
        kib = row["retained_activation_bytes"] / 1024
        per = row["activation_bytes_per_iteration"] / 1024
        print(f"{row['mode']:18}{kib:14.1f} KiB{per:11.1f} KiB{row['retained_graph_nodes']:8}")
    print(f"\nsecond backward: {second_backward_is_refused()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
