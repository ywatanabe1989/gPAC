#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUDA-Graphs feasibility probe for gPAC's per-permutation surrogate cost.

Question
--------
Is `ModulationIndex.compute_surrogates`'s per-permutation MI-aggregation
OVERHEAD-bound (kernel-launch / Python-dispatch gaps that CUDA Graphs would
remove) or COMPUTE-bound (the `mul` + `sum` aggregation itself)?

DECISION (this probe, on a dedicated + verified-empty H100 80GB, torch
2.2.2+cu121, seed 42): **COMPUTE-BOUND -> CUDA Graphs NOT implemented.**
The per-perm aggregation is >99.9% GPU kernel time; the launch/Python gap that
CUDA Graphs would remove is ~0% of the per-perm wall, so capturing the
aggregation in a CUDAGraph and replaying it gives no measurable speedup (and is
bit-identical to the eager loop). gPAC's production hot path
(`src/gpac/core/_ModulationIndex.py`) is therefore left UNCHANGED (no
reproducibility/VRAM/numerical impact).

Measured split (16ch x 25x25, n_perm=200), production NeuroVista config
(seq_len=24000, fp32): WALL/perm 19.29 ms ~= GPU-busy/perm 19.30 ms ->
overhead -0.05%; `cudaLaunchKernel` CPU ~0.06 ms/perm; the wall is
`aten::mul` (13.1 ms, the weights*amp broadcast) + `aten::sum` (6.5 ms). The
6.32x gPAC-vs-TensorPAC headline is a different config (seq_len=2048, fp16,
~2.4 ms/perm); the overhead-vs-compute verdict is the same at every config
this probe sweeps (see the printed table). This probe is intentionally NOT
wired into the production path; it exists to justify (and let a reviewer
reproduce) that decision.

What it measures (on the current CUDA device), per condition (seq_len, fp16),
16ch x 25x25, n_perm=200:
  * WALL/perm   : 200 perm bodies back-to-back, single final sync
  * GPU-busy/perm: CUDA events around each body (pure kernel time)
  * OVERHEAD/perm = WALL - GPU-busy
  * CUDA-Graphs replay/perm vs eager/perm, plus bit-equivalence (max|Δ|) of the
    graph output vs eager.

The per-perm body is EXACTLY the inner loop of
``ModulationIndex.compute_surrogates``:
    amp_sw = torch.cat([amp[..., cut:], amp[..., :cut]], dim=-1)
    mi, _  = self._mi_from_weights(weights, amp_sw)

Run on a DEDICATED, EMPTY GPU for meaningful launch-vs-kernel timing:
    PYTHONPATH=src python benchmark/surrogate_cuda_graphs/cuda_graphs_surrogate_probe.py

Reproducibility: seed=42; pass --deterministic to also pin the determinism
flags via ``gpac.set_deterministic``.
"""

import argparse
import json
import time

import numpy as np
import torch


def _build_inputs(seq_len, fp16, seed=42):
    from gpac import PAC

    dev = "cuda"
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((1, 16, seq_len)).astype("float32")).to(
        dev
    )
    model = (
        PAC(
            seq_len=seq_len,
            fs=400,
            pha_range_hz=(2, 30),
            amp_range_hz=(60, 180),
            pha_n_bands=25,
            amp_n_bands=25,
            n_perm=200,
            fp16=fp16,
            random_seed=seed,
            device_ids=[0],
            compile_mode=True,
        )
        .to(dev)
        .eval()
    )
    if fp16:
        x = x.half()
    x4 = x.unsqueeze(2)
    b, c, s, _ = x4.shape
    with torch.no_grad():
        xf = model.bandpass(x4.reshape(-1, seq_len))
        info = model.bandpass.info
        n_pha, n_amp = info["pha_n_bands"], info["amp_n_bands"]
        xf = xf.reshape(b, c, s, -1, seq_len)
        ph = model.hilbert(xf[:, :, :, :n_pha, :].reshape(-1, seq_len))
        am = model.hilbert(xf[:, :, :, n_pha:, :].reshape(-1, seq_len))
        phase = (
            ph[..., 0]
            .reshape(b, c, s, n_pha, seq_len)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        amplitude = (
            am[..., 1]
            .reshape(b, c, s, n_amp, seq_len)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        weights = model.mi_calculator._compute_weights(phase)
    return model, amplitude, weights


def _split_and_graph(seq_len, fp16, n_perm=200, seed=42):
    model, amplitude, weights = _build_inputs(seq_len, fp16, seed=seed)
    mi = model.mi_calculator
    time_dim = amplitude.shape[-1]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    cuts = torch.randint(1, time_dim, (n_perm,), device="cuda", generator=gen).tolist()

    def body(cut):
        amp_sw = torch.cat([amplitude[..., cut:], amplitude[..., :cut]], dim=-1)
        smi, _ = mi._mi_from_weights(weights, amp_sw, compute_distributions=False)
        return smi

    for i in range(10):
        _ = body(cuts[i])
    torch.cuda.synchronize()

    # WALL/perm (back-to-back)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_perm):
        _ = body(cuts[i])
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / n_perm * 1e3

    # GPU-busy/perm (cuda events)
    st = [torch.cuda.Event(enable_timing=True) for _ in range(n_perm)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(n_perm)]
    torch.cuda.synchronize()
    for i in range(n_perm):
        st[i].record()
        _ = body(cuts[i])
        en[i].record()
    torch.cuda.synchronize()
    busy = float(np.mean([st[i].elapsed_time(en[i]) for i in range(n_perm)]))

    # ---- CUDA-Graphs capture/replay over the aggregation ----
    # Precompute swapped amps so eager and graph both time ONLY the
    # _mi_from_weights aggregation (the part proposed for capture).
    swapped = [
        torch.cat([amplitude[..., c:], amplitude[..., :c]], dim=-1).contiguous()
        for c in cuts
    ]
    torch.cuda.synchronize()

    def agg(amp_in):
        smi, _ = mi._mi_from_weights(weights, amp_in, compute_distributions=False)
        return smi

    for i in range(10):
        _ = agg(swapped[i])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    eager_ref = None
    for i in range(n_perm):
        eager_ref = agg(swapped[i])
    torch.cuda.synchronize()
    eager_agg = (time.perf_counter() - t0) / n_perm * 1e3
    eager_ref = eager_ref.detach().clone()

    static_in = swapped[0].clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = agg(static_in)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph_ok, graph_err, graph_agg, max_abs = True, "", float("nan"), float("nan")
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = agg(static_in)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n_perm):
            static_in.copy_(swapped[i])
            g.replay()
        torch.cuda.synchronize()
        graph_agg = (time.perf_counter() - t0) / n_perm * 1e3
        max_abs = float(
            (static_out.detach().float() - eager_ref.float()).abs().max().item()
        )
    except Exception as exc:  # pragma: no cover - hardware dependent
        graph_ok, graph_err = False, repr(exc)

    return {
        "seq_len": seq_len,
        "fp16": fp16,
        "n_perm": n_perm,
        "time_dim": int(time_dim),
        "wall_per_perm_ms": wall,
        "gpu_busy_per_perm_ms": busy,
        "overhead_per_perm_ms": wall - busy,
        "overhead_frac": (wall - busy) / wall,
        "eager_agg_per_perm_ms": eager_agg,
        "graph_replay_per_perm_ms": graph_agg,
        "graph_speedup_x": (eager_agg / graph_agg)
        if graph_agg == graph_agg
        else float("nan"),
        "graph_vs_eager_max_abs_diff": max_abs,
        "graph_capture_ok": graph_ok,
        "graph_error": graph_err,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This probe requires a CUDA GPU.")
    if args.deterministic:
        from gpac import set_deterministic

        set_deterministic(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dev = torch.cuda.get_device_name(0)
    print(f"device: {dev}  torch {torch.__version__}  seed {args.seed}")
    rows = []
    for seq_len, fp16 in [(2048, True), (2048, False), (24000, True), (24000, False)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        r = _split_and_graph(seq_len, fp16, seed=args.seed)
        rows.append(r)
        print(
            f"seq={r['seq_len']:>5} fp16={str(r['fp16']):>5} | "
            f"wall/perm {r['wall_per_perm_ms']:8.4f} ms | "
            f"gpu-busy {r['gpu_busy_per_perm_ms']:8.4f} ms | "
            f"overhead {r['overhead_per_perm_ms']:+7.4f} ms "
            f"({r['overhead_frac'] * 100:5.2f}%) | "
            f"graph replay {r['graph_replay_per_perm_ms']:8.4f} ms "
            f"({r['graph_speedup_x']:.3f}x, max|d|={r['graph_vs_eager_max_abs_diff']:.2e})"
        )
    maxfrac = max(abs(r["overhead_frac"]) for r in rows)
    verdict = "COMPUTE-BOUND" if maxfrac < 0.10 else "OVERHEAD-BOUND/MIXED"
    print(
        f"max |overhead fraction| = {maxfrac * 100:.2f}% -> {verdict} "
        f"(CUDA Graphs {'NOT worth implementing' if verdict == 'COMPUTE-BOUND' else 'worth a look'})"
    )
    print("RESULT_JSON=" + json.dumps({"device": dev, "rows": rows}, default=str))


if __name__ == "__main__":
    main()
