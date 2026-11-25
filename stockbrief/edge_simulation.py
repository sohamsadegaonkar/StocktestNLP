"""Deterministic discrete-time cooperative edge simulation. No actual network or ML."""
from math import ceil

TICK_MS = 10
LIMIT_MS = 120_000
LEASE_MS = 200
WORKER_OUTAGE = (200, 1300)
SOURCE_OUTAGE = (200, 1200)
PROFILES = {"laptop": 1.0, "gateway": 0.65, "worker": 1.8}
ARCHITECTURES = ("fifo", "adaptive", "cooperative")


def _round_tick(ms):
    return int(ceil(ms / TICK_MS)) * TICK_MS


def workload(count, burst):
    jobs = []
    symbols = ("TCS", "RELIANCE", "HDFCBANK", "INFY", "TMPV", "INFY")
    categories = ("Routine", "Operations", "Regulatory", "Earnings", "Pricing", "Correction")
    for i in range(count):
        kind = i % len(symbols)
        jobs.append({"id": i+1, "ticker": symbols[kind], "event": categories[kind],
                     "relevant": kind >= 3, "correction": kind == 5,
                     "published_ms": 0 if burst else i*40,
                     "work_ms": (120, 160, 220, 240, 180, 260)[kind]})
    return jobs


def _down(at, window):
    return window is not None and window[0] <= at < window[1]


def simulate(jobs, architecture, latency, scenario):
    nodes = {name: {"speed": speed, "active": None, "completed": 0}
             for name, speed in PROFILES.items() if architecture == "cooperative" or name == "laptop"}
    worker_window = WORKER_OUTAGE if scenario in {"worker_outage", "combined"} and "worker" in nodes else None
    source_window = SOURCE_OUTAGE if scenario in {"source_outage", "combined"} else None
    pending, received, completed, trace, attempts, late = {}, set(), {}, [], [], []
    packets = []
    for job in jobs:
        published = job["published_ms"]
        delivered = source_window[1] if _down(published, source_window) else published
        packets.append((delivered, job["id"], job))
        if job["id"] % 5 == 0:
            duplicate_at = delivered + 50
            if _down(duplicate_at, source_window):
                duplicate_at = source_window[1]
            packets.append((duplicate_at, job["id"], job))
    packets.sort(key=lambda p: (p[0], p[1]))
    cursor = retries = duplicates = stale = network_bytes = 0

    def log(t, event, job=None, node=None, **extra):
        trace.append({"time_ms": t, "event": event, "job": job, "node": node, **extra})

    for t in range(0, LIMIT_MS+1, TICK_MS):
        for label, window in (("source", source_window), ("worker", worker_window)):
            if window and t in window:
                log(t, label + ("_down" if t == window[0] else "_recovered"),
                    node="worker" if label == "worker" else None)
        # Source packets are independent of inference: already received work stays available.
        while cursor < len(packets) and packets[cursor][0] <= t:
            _, jid, job = packets[cursor]
            cursor += 1
            if jid in received:
                duplicates += 1
                log(t, "duplicate_packet", jid)
                continue
            received.add(jid)
            pending[jid] = {**job, "received_ms": t, "ready_ms": t}
            log(t, "received", jid)
        # Lease invalidation models a worker link loss; stale results never commit.
        for name, node in nodes.items():
            active = node["active"]
            if not active:
                continue
            job, attempt = active
            if name == "worker" and _down(t, worker_window):
                attempt.update(end_ms=t, state="lease_lost")
                node["active"] = None
                pending[job["id"]] = {**job, "ready_ms": t+LEASE_MS}
                retries += 1
                log(t, "lease_lost", job["id"], name, retry_at_ms=t+LEASE_MS)
                # Model a delayed result from the invalidated attempt after reconnect.
                late.append((_round_tick(max(worker_window[1], attempt["planned_finish_ms"])+latency), job["id"], name))
            elif attempt["planned_finish_ms"] <= t:
                attempt.update(end_ms=t, state="completed")
                node["active"] = None
                if job["id"] in completed:
                    stale += 1
                    log(t, "stale_result", job["id"], name)
                else:
                    completed[job["id"]] = {"job": job["id"], "ticker": job["ticker"],
                        "event": job["event"], "relevant": job["relevant"],
                        "published_ms": job["published_ms"], "received_ms": job["received_ms"],
                        "completed_ms": t, "latency_ms": t-job["published_ms"], "node": name}
                    node["completed"] += 1
                    if name != "laptop":
                        network_bytes += 512  # illustrative result payload
                    log(t, "completed", job["id"], name)
        for due, jid, name in late[:]:
            if due <= t:
                late.remove((due, jid, name))
                stale += 1
                network_bytes += 512
                log(t, "stale_result", jid, name)
        ready = [j for j in pending.values() if j["ready_ms"] <= t]
        if architecture == "fifo":
            ready.sort(key=lambda j: (j["received_ms"], j["id"]))
        else:
            ready.sort(key=lambda j: (-(4*int(j["relevant"]) + 3*int(j["correction"]) +
                                         (t-j["received_ms"])/600), j["received_ms"], j["id"]))
        for job in ready:
            free = [(name, node) for name, node in nodes.items() if node["active"] is None
                    and not (name == "worker" and _down(t, worker_window))]
            if not free:
                break
            def duration(pair):
                name, node = pair
                return _round_tick(job["work_ms"]/node["speed"] + (2*latency if name != "laptop" else 0))
            name, node = min(free, key=lambda pair: (duration(pair), pair[0]))
            finish = t+duration((name, node))
            attempt = {"job": job["id"], "node": name, "start_ms": t,
                       "planned_finish_ms": finish, "end_ms": None, "state": "active"}
            attempts.append(attempt)
            node["active"] = (job, attempt)
            del pending[job["id"]]
            if name != "laptop":
                network_bytes += 2048  # illustrative public document payload
            log(t, "assigned", job["id"], name, planned_finish_ms=finish)
        if len(completed) == len(jobs) and cursor == len(packets) and not late:
            break

    results = sorted(completed.values(), key=lambda r: (r["completed_ms"], r["job"]))
    relevant = [r["completed_ms"] for r in results if r["relevant"]]
    latencies = sorted(r["latency_ms"] for r in results)
    return {"architecture": architecture, "completed": len(results), "total": len(jobs),
            "first_relevant_ms": min(relevant) if relevant else None,
            "all_relevant_ms": max(relevant) if len(relevant) == sum(j["relevant"] for j in jobs) else None,
            "makespan_ms": max((r["completed_ms"] for r in results), default=None),
            "p95_latency_ms": latencies[ceil(.95*len(latencies))-1] if latencies else None,
            "duplicates": duplicates, "lease_retries": retries, "stale_results": stale,
            "modeled_network_bytes": network_bytes, "end_ms": t,
            "nodes": [{"name": name, "speed": node["speed"], "completed": node["completed"]} for name, node in nodes.items()],
            "results": results, "attempts": attempts, "trace": trace,
            "source_outage": source_window, "worker_outage": worker_window}


def experiment(count=24, burst=True, latency=30, scenario="combined"):
    if type(count) is not int or not 6 <= count <= 60 or type(latency) is not int or not 0 <= latency <= 300:
        raise ValueError("Use 6–60 jobs and 0–300 ms one-way delay.")
    if type(burst) is not bool or scenario not in {"normal", "worker_outage", "source_outage", "combined"}:
        raise ValueError("Invalid workload or failure scenario.")
    jobs = workload(count, burst)
    return {"scope": "Virtual-time simulation only; assumed device speeds, work durations and network delays. No real network, inference, energy or price measurements.",
            "parameters": {"count": count, "burst": burst, "latency_ms": latency, "scenario": scenario,
                           "tick_ms": TICK_MS, "lease_retry_ms": LEASE_MS, "speeds": PROFILES,
                           "assumed_request_bytes": 2048, "assumed_result_bytes": 512},
            "workload": jobs,
            "runs": {a: simulate(jobs, a, latency, scenario) for a in ARCHITECTURES}}
