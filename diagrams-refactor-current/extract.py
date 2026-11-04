"""Per-CB timing for iteration 1 on both ranks -> data.json (ms after rank 0's
SpyreCommsWorkSchedule(id:2): new; both ranks share the host clock)."""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
LOGS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "logs"

TS = re.compile(r"\d\d\.\d\d\.\d{4} \d\d:\d\d:(\d\d\.\d{6})")


def t_of(line):
    m = TS.search(line)
    return float(m.group(1)) if m else None


def parse(path):
    lines = open(path).read().splitlines()
    beg = next(i for i, l in enumerate(lines) if "### ITER 1 BEGIN" in l)
    end = next(i for i, l in enumerate(lines) if "### ITER 1 END" in l)
    seg = lines[beg : end + 1]
    cbs, order = {}, []
    label = cur = None
    bar = None  # (enter, exit, total, kind) of the last barrier
    sched = None  # first scheduler line of the op being scheduled
    send = None
    pipe = None  # pipeline named by the ResponseWorker before it sends the CB
    last_resp = None
    ev = {}
    for l in seg:
        t = t_of(l)
        if t is None:
            pass
        if "STRICT_ORDERING" in l or "RuntimeOperationCompute" in l or "submitDma" in l:
            if sched is None:
                sched = t
        m = re.search(
            r"issueBarrier ENTER: op_type=\d+ stream_id=\d+ total_fences=(\d+) \((.*)\)",
            l,
        )
        if m:
            bar = [t, None, int(m.group(1)), m.group(2)]
            if sched is None:
                sched = t
        if "issueBarrier EXIT" in l and bar:
            bar[1] = t
        m = re.search(r"Pipeline: PipelineId\((\w+)\)", l)
        if m:
            pipe = m.group(1)
        if "Sending CB" in l:
            send = t
        if l.startswith("CB label "):
            label = l[len("CB label ") :]
        m = re.search(r"ControlSection\(contextId:\d+, tag:(\d+).*barrier:(\w+)", l)
        if m:
            cur = int(m.group(1))
            cbs[cur] = dict(
                tag=cur,
                label=label,
                barrier_bit=m.group(2) == "true",
                sched=sched,
                bar=bar,
                send=send,
                pipeline=pipe,
            )
            order.append(cur)
            sched, bar = None, None
        m = re.search(r"Tapping queue to (\d+)", l)
        if m and cur is not None and "door" not in cbs[cur]:
            cbs[cur]["queue"] = int(m.group(1))
            cbs[cur]["door"] = t
        if "Got response" in l:
            last_resp = t
        m = re.search(r"invoking callback for PendingRequest ID (\d+)", l)
        if m:
            ev.setdefault(int(m.group(1)), {})["cb"] = t
            ev[int(m.group(1))]["resp"] = last_resp
        m = re.search(r"callback completed for PendingRequest ID (\d+)", l)
        if m:
            ev.setdefault(int(m.group(1)), {})["done"] = t
        for key in (
            "SpyreCommsWorkSchedule",
            "HDMA P2P Protocol: Begin",
            "Launched WorkSchedule",
            "synchronize()",
            "Destroying",
            "Block assigned",
            "Block freed",
            "LaunchAll: WorkSchedule",
        ):
            if key in l:
                ev.setdefault("log", []).append((t, l[l.find("] ", 60) + 2 :][:140]))
    # PendingRequest IDs are issued in CB order
    ids = sorted(k for k in ev if isinstance(k, int))
    for tag, rid in zip(order, ids):
        cbs[tag].update(ev[rid])
    return cbs, order, ev.get("log", [])


out = {}
t0 = None
for r in (0, 1):
    cbs, order, log = parse(LOGS / f"output-at-rank-{r}.txt")
    if t0 is None:
        t0 = next(t for t, l in log if ": new" in l)
    ms = lambda v: None if v is None else round((v - t0) * 1000, 3)
    rows = []
    for i, tag in enumerate(order):
        c = cbs[tag]
        b = c["bar"]
        rows.append(
            dict(
                i=i,
                tag=tag,
                label=c["label"],
                queue=c.get("queue"),
                pipeline=c["pipeline"],
                barrier_bit=c["barrier_bit"],
                sched=ms(c["sched"]),
                send=ms(c["send"]),
                bar_enter=ms(b[0]) if b else None,
                bar_exit=ms(b[1]) if b else None,
                bar_fences=b[2] if b else None,
                bar_kinds=b[3] if b else None,
                door=ms(c.get("door")),
                resp=ms(c.get("resp")),
                cb=ms(c.get("cb")),
                done=ms(c.get("done")),
            )
        )
    out[f"r{r}"] = dict(cbs=rows, log=[(ms(t), l) for t, l in log])
json.dump(out, open(HERE / "data.json", "w"), indent=1)
for r in ("r0", "r1"):
    print("==", r)
    for c in out[r]["cbs"]:
        print(
            f"{c['i']:2d} {c['tag']:6d} {c['pipeline'][:7]:7} q{c['queue']:<3} {'B' if c['barrier_bit'] else '-'} "
            f"sched {c['sched']} bar {c['bar_enter']}->{c['bar_exit']} ({c['bar_fences']}) "
            f"send {c['send']} door {c['door']} resp {c['resp']} cb {c['cb']} done {c['done']}  "
            f"{(c['label'] or '')[-30:]}"
        )
    for t, l in out[r]["log"]:
        print("   ", t, l[:110])
