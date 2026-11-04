# Builds lifecycle.svg, timeline.svg and the CB table from data.json (see
# extract.py) and splices them into index.src.html -> index.html.
# Rank 0, iteration 1; times in ms after rank 0's
# SpyreCommsWorkSchedule(id:2): new.
#
# Everything measured comes from data.json. What is written here by hand is
# naming (NAME/LONG/DESC), the push times the log does not pin (PUSH), which
# callback releases which barrier (WAITS_ON, checked against the log), and the
# host-side milestones of the Python thread (PY, read off the log).
import json
from pathlib import Path

HERE = Path(__file__).parent
D = json.loads((HERE / "data.json").read_text())
# CB roles by position in iteration 1. Tags and queue numbers are assigned at
# run time and differ between runs (105..119 in one, 27..41 in another), so
# everything below is keyed by position k and the tags are read from the data.
ROLES = [
    "H2D mul correction",
    "fused_mul",
    "D2H 256 B",
    "RDMA wait",
    "H2D 512 B",
    "sdsc_add_0",
    "D2H 128 B",
    "RDMA wait",
    "comms_ws_end",
    "H2D add_1 correction",
    "fused_add_1 (y+1)",
    "H2D add_2 correction",
    "fused_add_2",
    "D2H 256 B to CPU",
]
LABEL_KEYS = [
    "dma_h2d_0",
    "fused_wait_tensor",
    "data_mv_d2h",
    "hwsid_wait",
    "data_mv_h2d",
    "sdsc_add_0",
    "dma_d2h_compute_0",
    "p2p_rdma_wait",
    "comms_ws_end",
    "dma_h2d_0",
    "fused_add",
    "dma_h2d_0",
    "fused_add",
    "dma_d2h_0",
]
N = len(ROLES)  # the 15th CB is the driver's own x.to("cpu") check
R0 = D["r0"]["cbs"][:N]
R1 = D["r1"]["cbs"][:N]
for R in (R0, R1):
    got = [c["label"] or "" for c in R]
    assert len(D["r0"]["cbs"]) == N + 1 and all(
        k in l for k, l in zip(LABEL_KEYS, got)
    ), f"CB sequence changed; update ROLES/LONG/DESC/WAITS_ON: {got}"
TAG = [c["tag"] for c in R0]  # rank 0 tag of each position
DRIVER_TAG = D["r0"]["cbs"][N]["tag"]
BY = {k: c for k, c in enumerate(R0)}
T = lambda k: TAG[k]

COLL = range(2, 9)  # positions of the collective CBs
own = lambda k: "cm" if k in COLL else "acc"
NAME = ROLES
LONG = [
    "H2D 256 B · mul program correction",
    "fused_mul compute (x * 2 → buf0)",
    "D2H 256 B · own contribution to staging",
    "p2p_rdma_wait (hwsid) · first peer mask",
    "H2D 512 B · both halves",
    "sdsc_add_0 compute",
    "D2H 128 B · seg 3 message",
    "p2p_rdma_wait · second peer mask",
    "comms_ws_end · barrier",
    "H2D 256 B · add_1 correction",
    "fused_add_1 compute (y + 1 → LX)",
    "H2D 256 B · add_2 correction",
    "fused_add_2 compute (buf0 + z)",
    "D2H 256 B · result to CPU",
]
# push time (ms) and whether the log pins it. LaunchAll appends the collective
# ops while it accumulates (0.81-0.91 ms); none of the pushes is logged.
PUSH = {
    0: (-0.03, False),
    1: (-0.03, False),
    9: (1.30, False),
    10: (1.30, False),
    11: (1.50, False),
    12: (1.50, False),
    13: (1.72, False),
}
for t in COLL:
    PUSH[t] = (0.81, False)
# op whose callback resolves the (last) fence a barrier is blocked on
WAITS_ON = {1: 0, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12}
for k, src in WAITS_ON.items():  # sanity: the release follows the callback
    assert BY[k]["bar_fences"] >= 1 and BY[src]["cb"] <= BY[k]["bar_exit"], k


# Python thread milestones on rank 0 (ms), from the log
def _log_t(key, nth=0):
    hits = [t for t, l in D["r0"]["log"] if key in l]
    return hits[nth]


PY = dict(
    slot=_log_t("acquired global slot"),
    launchall=_log_t("LaunchAll: WorkSchedule"),
    ar_ret=_log_t("Launched WorkSchedule"),
    sync_in=_log_t("synchronize() device=0 waiting"),
    sync_out=_log_t("synchronize() device=0 completed"),
)
R1_NEW = next(t for t, l in D["r1"]["log"] if ": new" in l)
BIG = max(WAITS_ON, key=lambda k: BY[k]["bar_fences"])  # the barrier before y + 1
END = max(D["r0"]["cbs"][N]["done"], PY["sync_out"])  # last thing on rank 0
START = min(c["door"] for c in R1) if R1[0]["door"] < 0 else 0


def ticks(lo, hi, step=0.5):
    k0 = int(lo // step)
    return [k * step for k in range(k0, int(hi // step) + 1)]


def svg_lifecycle():
    W = 960
    T0, T1 = -0.15, END + 0.35
    X0 = 214
    PX = (W - 20 - X0) / (T1 - T0)
    x = lambda t: X0 + (t - T0) * PX
    ROW0, RH = 64, 24
    G1, G2 = 2, 2 + len(COLL)  # first collective row, first graph row after it
    ry = lambda i: ROW0 + i * RH + (10 if i >= G1 else 0) + (10 if i >= G2 else 0)
    o = []

    def rect(a, b, y, h, cls, title=None):
        w = max(x(b) - x(a), 1.5)
        t = f"<title>{title}</title>" if title else ""
        o.append(
            f'<rect class="{cls}" x="{x(a):.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" rx="1.5">{t}</rect>'
        )

    last = ry(len(R0) - 1) + RH
    LANES = [
        ("Python thread", "run · push · sync"),
        ("scheduler thread", "pops ops, barriers"),
        ("ResponseWorker", "doorbell, callbacks"),
        ("device", "doorbell → response"),
    ]
    LY0, LH = last + 30, 38
    bottom = LY0 + len(LANES) * LH
    for t in ticks(0, T1):
        o.append(
            f'<line class="grid" x1="{x(t):.1f}" y1="44" x2="{x(t):.1f}" y2="{bottom}"/>'
        )
        o.append(
            f'<text class="time" x="{x(t):.1f}" y="34" text-anchor="middle">{t:g}</text>'
        )
    o.append('<text class="time" x="12" y="34">ms</text>')
    o.append('<text class="t2" x="12" y="54">stream_id 1 · op</text>')

    for i, c in enumerate(R0):
        k = i
        tag = T(k)
        y = ry(i)
        cy = y + RH / 2
        bh = 12
        by = cy - bh / 2
        o.append(
            f'<line class="lane" x1="12" y1="{y + RH:.1f}" x2="{W - 12}" y2="{y + RH:.1f}"/>'
        )
        o.append(f'<text class="tag {own(k)}" x="12" y="{cy + 4:.1f}">{tag}</text>')
        o.append(f'<text class="lab" x="46" y="{cy + 4:.1f}">{NAME[k]}</text>')
        enq, exact = PUSH[k]
        rect(
            enq,
            c["sched"],
            by,
            bh,
            "st q" + ("" if exact else " fuzzy"),
            f"{tag}: in the stream deque from {'' if exact else '≈'}{enq:.2f} until popped at {c['sched']:.3f}",
        )
        rect(
            c["sched"],
            c["send"],
            by,
            bh,
            "st s",
            f"{tag}: scheduler thread {c['sched']:.3f} → handoff {c['send']:.3f}",
        )
        if k in WAITS_ON:
            rect(
                c["bar_enter"],
                c["bar_exit"],
                by,
                bh,
                "st b",
                f"{tag}: issueBarrier waits {c['bar_enter']:.3f} → {c['bar_exit']:.3f} on {c['bar_fences']} fence(s), the last from op {T(WAITS_ON[k])}",
            )
        rect(
            c["send"],
            c["door"],
            by,
            bh,
            "st h",
            f"{tag}: handed to ResponseWorker {c['send']:.3f}, doorbell {c['door']:.3f}",
        )
        rect(
            c["door"],
            c["resp"],
            by,
            bh,
            f"bar {own(k)}",
            f"{tag}: on device {c['door']:.3f} → {c['resp']:.3f} ({(c['resp'] - c['door']) * 1000:.0f} µs)",
        )
        rect(
            c["resp"],
            c["done"],
            by,
            bh,
            f"bar {own(k)}w",
            f"{tag}: response {c['resp']:.3f}, callback {c['cb']:.3f} → {c['done']:.3f}",
        )
    for i in (G1, G2):
        o.append(
            f'<line class="lane strong" x1="12" y1="{ry(i) - 5:.1f}" x2="{W - 12}" y2="{ry(i) - 5:.1f}"/>'
        )

    for k, src in WAITS_ON.items():
        p = R0[src]
        c = R0[k]
        x1, y1 = x(p["cb"]), ry(src) + RH / 2 + 6
        x2, y2 = x(c["bar_exit"]), ry(k) + RH / 2 - 7
        o.append(
            f'<path class="edge bad" d="M{x1:.1f},{y1:.1f} C{x1:.1f},{y2 - 2:.1f} {x2:.1f},{y1 + 2:.1f} {x2:.1f},{y2:.1f}" marker-end="url(#fa)"/>'
        )

    def badge(n, t, i):
        cx, cy = x(t), ry(i) + RH / 2
        o.append(
            f'<circle class="badge" cx="{cx:.1f}" cy="{cy:.1f}" r="8"/><text class="badgen" x="{cx:.1f}" y="{cy + 3.8:.1f}" text-anchor="middle">{n}</text>'
        )

    badge(1, BY[1]["done"] + 0.12, 1)
    badge(2, BY[2]["door"] - 0.35, G1)
    badge(3, BY[4]["done"] + 0.15, G1 + 2)
    badge(4, BY[BIG]["bar_enter"] - 0.12, G2)
    badge(5, BY[13]["done"] + 0.15, N - 1)

    for j, (a, b) in enumerate(LANES):
        y = LY0 + j * LH
        o.append(
            f'<line class="lane" x1="12" y1="{y + LH:.1f}" x2="{W - 12}" y2="{y + LH:.1f}"/>'
        )
        o.append(
            f'<text class="t1" x="12" y="{y + 16:.1f}">{a}</text><text class="t2" x="12" y="{y + 31:.1f}">{b}</text>'
        )
    o.append(
        f'<line class="lane strong" x1="12" y1="{LY0 - 8:.1f}" x2="{W - 12}" y2="{LY0 - 8:.1f}"/>'
    )
    ly = lambda j: LY0 + j * LH + 13

    y = ly(0)
    rect(
        0,
        PY["launchall"],
        y,
        12,
        "bar ink",
        "allreduce_run: WorkSchedule new, allocDev, start(), global slot, LaunchAll",
    )
    rect(
        PY["launchall"],
        PY["ar_ret"],
        y,
        12,
        "bar cmw",
        f"allreduce_run: LaunchAll records and accumulates the ops; returns at {PY['ar_ret']:.3f}, after {T(2)}'s callback",
    )
    rect(
        PY["ar_ret"],
        PY["sync_in"],
        y,
        12,
        "bar ink",
        "fused_add_1.run, wait_work (returns at once), fused_add_2.run, .to('cpu')",
    )
    rect(
        PY["sync_in"],
        PY["sync_out"],
        y,
        12,
        "bar accw",
        "RuntimeStream::synchronize(): blocked until in_flight and queued reach 0",
    )
    o.append(
        f'<text class="lab cm inbar" x="{x(PY["launchall"]) + 6:.1f}" y="{y + 10:.1f}">allreduce_run</text>'
    )
    o.append(
        f'<text class="lab acc inbar" x="{x(PY["sync_in"]) + 6:.1f}" y="{y + 10:.1f}">synchronize() blocks</text>'
    )
    for t in sorted({p for p, _ in PUSH.values()}):
        o.append(
            f'<path class="pushm" d="M{x(t) - 4:.1f},{y - 1:.1f} L{x(t) + 4:.1f},{y - 1:.1f} L{x(t):.1f},{y - 8:.1f} z"/>'
        )
    y = ly(1)
    for c in R0:
        rect(c["sched"], c["send"], y, 12, "st s", f"{c['tag']}: scheduler thread")
        if R0.index(c) in WAITS_ON:
            rect(
                c["bar_enter"],
                c["bar_exit"],
                y,
                12,
                "st b",
                f"{c['tag']}: issueBarrier blocks the scheduler thread",
            )
    y = ly(2)
    for c in R0:
        rect(c["send"], c["door"], y, 12, "st h", f"{c['tag']}: send CB, ring doorbell")
        rect(
            c["resp"],
            c["done"],
            y,
            12,
            f"bar {own(R0.index(c))}w",
            f"{c['tag']}: response → callback done",
        )
    y = ly(3)
    for c in R0:
        rect(
            c["door"],
            c["resp"],
            y,
            12,
            f"bar {own(R0.index(c))}",
            f"{c['tag']}: on device",
        )

    b = BY[BIG]
    head = (
        f'<svg viewBox="0 0 {W} {bottom + 10}" role="img" aria-label="Life of each op on stream 1, rank 0, iteration 1. '
        f"The multiply's two ops run first and finish by {BY[1]['done']:.2f} ms. allreduce_run returns at {PY['ar_ret']:.2f} ms, "
        f"after LaunchAll has pushed seven collective ops; they run back to back on the COMPUTE queue until {BY[8]['done']:.2f} ms. "
        f"y + 1 is pushed behind them. Op {T(BIG)}, the first op of y + 1, blocks the scheduler thread on {b['bar_fences']} compute fences "
        f"from {b['bar_enter']:.2f} to {b['bar_exit']:.2f} ms. The graph then ping-pongs between DMAI and COMPUTE "
        f'and synchronize() returns at {PY["sync_out"]:.2f} ms.">'
        '<defs><marker id="fa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" class="head-bad"/></marker></defs>'
    )
    return head + "\n" + "\n".join(o) + "\n</svg>"


def svg_timeline():
    W, H = 960, 540
    T0, T1 = min(START, 0) - 0.15, END + 0.35
    X0 = 172
    S = (W - 16 - X0) / (T1 - T0)
    x = lambda t: X0 + (t - T0) * S
    o = []
    for t in ticks(T0 + 0.1, T1):
        o.append(
            f'<line class="grid" x1="{x(t):.1f}" y1="44" x2="{x(t):.1f}" y2="{H - 58}"/>'
        )
        o.append(
            f'<text class="time" x="{x(t):.1f}" y="34" text-anchor="middle">{t:g}</text>'
        )
    o.append('<text class="time" x="16" y="34">ms</text>')
    qs = lambda p: sorted(c["queue"] for c in R0 if c["pipeline"] == p)
    qc, qi, qo = qs("COMPUTE"), qs("ASYNC_DMAI"), qs("ASYNC_DMAO")
    rng = lambda q: f"queue {q[0]}" if q[0] == q[-1] else f"queue {q[0]} → {q[-1]}"
    lanes = [
        ("Python thread", "call() on rank 0", 52),
        ("spyre_comms", "WorkSchedule id 2", 98),
        ("flex scheduler", "host barrier waits", 144),
        ("COMPUTE", rng(qc), 190),
        ("ASYNC_DMAI", rng(qi), 262),
        ("ASYNC_DMAO", rng(qo), 308),
        ("Rank 1", "all three queues", 364),
    ]
    for a, b, y in lanes:
        o.append(
            f'<text class="t1{" cm" if a == "spyre_comms" else ""}" x="16" y="{y + 18}">{a}</text>'
        )
        o.append(f'<text class="t2" x="16" y="{y + 34}">{b}</text>')
    for y in (90, 136, 182, 254, 300, 346):
        o.append(f'<line class="lane" x1="16" y1="{y}" x2="{W - 16}" y2="{y}"/>')
    o.append(f'<line class="lane strong" x1="16" y1="352" x2="{W - 16}" y2="352"/>')

    def span(t0, t1, y, h, cls, title, label=None, lab_cls="lab", anchor="start"):
        o.append(
            f'<rect class="{cls}" x="{x(t0):.1f}" y="{y}" width="{max(x(t1) - x(t0), 2):.1f}" height="{h}" rx="2"><title>{title}</title></rect>'
        )
        if label:
            lx = x(t0) if anchor == "start" else (x(t0) + x(t1)) / 2
            o.append(
                f'<text class="{lab_cls}" x="{lx:.1f}" y="{y - 5}" text-anchor="{anchor}">{label}</text>'
            )

    y = 52
    span(
        0,
        PY["ar_ret"],
        y + 22,
        14,
        "bar cmw",
        f"allreduce_run: 0 → {PY['ar_ret']:.3f} ms",
        "allreduce_run",
        "lab cm",
    )
    span(
        PY["ar_ret"],
        PY["sync_in"],
        y + 22,
        14,
        "bar ink",
        f"y + 1, wait_work, add_2, .to(cpu): {PY['ar_ret']:.3f} → {PY['sync_in']:.3f} ms",
    )
    span(
        PY["sync_in"],
        PY["sync_out"],
        y + 22,
        14,
        "bar accw",
        f"RuntimeStream::synchronize(): {PY['sync_in']:.3f} → {PY['sync_out']:.3f} ms",
    )
    o.append(
        f'<text class="lab acc" x="{x(PY["sync_out"]):.1f}" y="{y + 17}" text-anchor="end">synchronize()</text>'
    )
    o.append(
        f'<path class="mark" d="M{x((PY["ar_ret"] + PY["sync_in"]) / 2):.1f},{y + 22} l5,7 l-5,7 l-5,-7 z"><title>wait_work(buf0): returns without blocking (default stream, TORCH_SPYRE_DIST_ENFORCE_WAIT unset); exact time not logged</title></path>'
    )

    y = 98
    span(
        0,
        PY["slot"],
        y + 22,
        14,
        "bar cm",
        f"WorkSchedule new → acquired global slot: 0 → {PY['slot']:.3f} ms",
    )
    span(
        PY["launchall"],
        PY["ar_ret"],
        y + 22,
        14,
        "bar cm",
        f"LaunchAll → Launched WorkSchedule: {PY['launchall']:.3f} → {PY['ar_ret']:.3f} ms",
        "LaunchAll → Launched",
        "lab cm",
    )
    o.append(
        f'<text class="lab cm" x="{x(PY["ar_ret"]) + 8:.1f}" y="{y + 33}">rank 1 started at {R1_NEW:+.2f}</text>'
    )

    y = 144
    for k, c in enumerate(R0):
        if k in WAITS_ON:
            big = k == BIG
            span(
                c["bar_enter"],
                c["bar_exit"],
                y + 22,
                14,
                "bar bad" if big else "bar ink",
                f"issueBarrier for {c['tag']}: {c['bar_enter']:.3f} → {c['bar_exit']:.3f} ms, {c['bar_fences']} fence(s)",
            )
            if big:
                o.append(
                    f'<text class="lab bad" x="{x(c["bar_enter"]) - 6:.1f}" y="{y + 33}" text-anchor="end">y + 1 waits on {c["bar_fences"]} compute fences</text>'
                )
    o.append(
        f'<text class="lab mut" x="{x(BY[1]["bar_exit"]) + 6:.1f}" y="{y + 33}">{T(1)}: {BY[1]["bar_fences"]} fences</text>'
    )

    def cb(c, ybar, labpos, peer=False):
        tag, pos = c["tag"], c["i"]
        k = own(pos)
        o.append(
            f'<rect class="bar {k}w" x="{x(c["door"]):.1f}" y="{ybar}" width="{x(c["done"]) - x(c["door"]):.1f}" height="12" rx="2"><title>{"Rank 1 " if peer else ""}tag {tag} · {LONG[pos]} · doorbell {c["door"]:.3f}, callback done {c["done"]:.3f} ms</title></rect>'
        )
        o.append(
            f'<rect class="bar {k}" x="{x(c["door"]):.1f}" y="{ybar}" width="{max(x(c["resp"]) - x(c["door"]), 2):.1f}" height="12" rx="2"><title>tag {tag}: on device {c["door"]:.3f} → {c["resp"]:.3f} ms</title></rect>'
        )
        ly = ybar - 4 if labpos == "up" else ybar + 23
        o.append(f'<text class="tag {k}" x="{x(c["door"]):.1f}" y="{ly}">{tag}</text>')

    ci = 0
    for c in R0:
        if c["pipeline"] == "ASYNC_DMAI":
            cb(c, 262 + 22, "up")
        elif c["pipeline"] == "ASYNC_DMAO":
            cb(c, 308 + 22, "up")
        else:
            up = ci % 2 == 0
            ci += 1
            cb(c, 190 + (20 if up else 38), "up" if up else "down")
    for j, c in enumerate(R1):
        up = j % 2 == 0
        cb(c, 364 + (20 if up else 38), "up" if up else "down", peer=True)

    y = H - 44
    for a, b, l, k in [
        (BY[0]["door"], BY[1]["done"], "x * 2", "acc"),
        (BY[2]["door"], BY[8]["done"], f"allreduce · {len(COLL)} CBs on COMPUTE", "cm"),
        (BY[9]["door"], BY[10]["done"], "y + 1", "acc"),
        (BY[11]["door"], BY[12]["done"], "+ z", "acc"),
        (BY[13]["door"], BY[13]["done"], "CPU", "acc"),
    ]:
        o.append(f'<path class="edge {k}" d="M{x(a):.1f},{y - 6} v6 H{x(b):.1f} v-6"/>')
        o.append(
            f'<text class="lab {k}" x="{(x(a) + x(b)) / 2:.1f}" y="{y + 16}" text-anchor="middle">{l}</text>'
        )
    r1 = R1
    rank1 = (
        f"starting {-R1_NEW:.2f} ms earlier"
        if R1_NEW < 0
        else f"starting {R1_NEW:.2f} ms later"
    )
    who = "rank 0" if R1_NEW < 0 else "nothing"
    head = (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="To-scale timeline of iteration 1. On rank 0 the multiply runs '
        f"from {BY[0]['door']:.2f} to {BY[1]['done']:.2f} ms. The {len(COLL)} collective control blocks go to the COMPUTE queue "
        f"between {BY[2]['door']:.2f} and {BY[8]['done']:.2f} ms. "
        f"y + 1 starts at {BY[9]['door']:.2f} ms, after a host barrier on {BY[BIG]['bar_fences']} compute fences, then the final add "
        f"and the copy to the CPU finish at {BY[13]['done']:.2f} ms. Rank 1 runs the same {len(R1)} control blocks, {rank1}; "
        f"its two RDMA waits take {(r1[3]['resp'] - r1[3]['door']) * 1000:.0f} and "
        f'{(r1[7]["resp"] - r1[7]["door"]) * 1000:.0f} µs.">'
    )
    return head + "\n" + "\n".join(o) + "\n</svg>"


GROUPS = {
    0: "sdsc_fused_mul_0.run(x, 2.0, buf0)",
    2: "allreduce_run(buf0) · WorkSchedule 2 · <code>AllReduce_Offload_AllGatherSum</code>",
    9: "sdsc_fused_add_1.run(y, 1.0) · z stays in LX",
    11: 'sdsc_fused_add_2.run(buf0, buf6) and .to("cpu")',
}
# Addresses below are from the run in logs/ (18:58 UTC, 4 Nov); they are
# stable across runs of this driver, but re-check them after a runtime change.
DESC = {
    0: "256 B program correction into seg 7 at <code>0x1c00000000</code>, through the runtime-mapped page <code>0x400005445000</code>.",
    1: "The multiply, booted from <code>0x1c00001080</code>. It writes <code>buf0</code> in seg 2. The kernel is named after <code>wait_tensor</code> because its provenance <code>aten_op</code> is the wait node.",
    2: "DMAO 256 B from <code>buf0</code> (seg 2 +0x0) to host staging <code>0x400002400000</code>, slot 0. Rank 0's contribution.",
    3: "RDMA10 wait, <code>list_bitmask 32</code>, <code>sync: true</code>. Holds the queue until rank 1's half is in staging. Rank 1 waits on mask 16 here.",
    4: "DMAI 512 B: the whole staging buffer into the receive buffer <code>temp_OFF_R0_T0</code> at seg 2 +0x100.",
    5: "The reduction kernel. Reads <code>xlat[0]</code> <code>0x800000180</code> and <code>xlat[1]</code> <code>0x800000280</code> (the two halves) and writes <code>xlat[2]</code> <code>0x800000080</code>, which is <code>buf0</code>. The 1,152 B program is already on the device.",
    6: "DMAO 128 B from seg 3 +0x80 to the message buffer <code>0x400005404000</code>.",
    7: "RDMA10 wait, <code>list_bitmask 1</code>. Rank 1 waits on mask 2.",
    8: "Empty CB with the barrier bit. Its response marks the end of the schedule.",
    9: f"256 B program correction at <code>0x1c00012900</code>. Its host barrier waited on {BY[9]['bar_fences']} compute fences, {BY[9]['bar_enter']:.2f} → {BY[9]['bar_exit']:.2f} ms.",
    10: "<code>y + 1</code>. Writes <code>z</code> to the LX scratchpad, not to HBM.",
    11: "256 B program correction at <code>0x1c00025200</code>.",
    12: "<code>buf0 + z</code>: reads the reduced <code>buf0</code> from HBM and <code>z</code> from LX, and writes <code>buf6</code> in seg 5.",
    13: f"256 B result from seg 5 to the CPU tensor. <code>synchronize()</code> returns at {PY['sync_out']:.3f} ms.",
}


def short(label):
    label = label.split("/")[-2] + "/bundle.mlir" if "bundle.mlir" in label else label
    label = label.replace("[WS 2.0] ", "").replace("spyre_kernel_v1_", "").strip()
    return label if len(label) <= 28 else label[:24] + "…" + label[-3:]


def cb_table():
    rows = []
    for k, c in enumerate(R0):
        if k in GROUPS:
            rows.append(
                f'          <tr class="group"><td colspan="7">{GROUPS[k]}</td></tr>'
            )
        dev = round((c["resp"] - c["door"]) * 1000)
        b = "yes" if c["barrier_bit"] else "no"
        rows.append(
            f'          <tr class="{"cm" if k in COLL else "ac"}"><td>{c["tag"]}</td><td class="lbl">{short(c["label"])}</td>'
            f'<td>{c["pipeline"]} {c["queue"]}</td><td class="{b}">{b}</td><td>{DESC[k]}</td>'
            f'<td class="num">{c["door"]:.3f}</td><td class="num">{dev:,} µs</td></tr>'
        )
    return "\n".join(rows)


if __name__ == "__main__":
    src = (HERE / "index.src.html").read_text()
    lc, tl = svg_lifecycle(), svg_timeline()
    (HERE / "lifecycle.svg").write_text(lc)
    (HERE / "timeline.svg").write_text(tl)
    out = (
        src.replace("<!--LIFECYCLE-->", lc)
        .replace("<!--TIMELINE-->", tl)
        .replace("<!--CBTABLE-->", cb_table())
    )
    assert "<!--" + "LIFECYCLE" not in out
    (HERE / "index.html").write_text(out)
    print("ok", len(out))
