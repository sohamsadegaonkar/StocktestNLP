# Cooperative edge network — simulation only

Open local Edge Lab and run the controlled simulation. Configure 6–60
synthetic financial announcements, simultaneous or streaming arrival,
0–300 ms one-way network delay, and no failure / worker loss / source
outage / combined. Replay the trace via virtual clock. No LAN sockets,
physical hardware, accounts, or remote inference are needed.

All architectures use the same synthetic job list and assumed work costs.
The single laptop FIFO and adaptive baselines each have one worker; the
cooperative policy adds a virtual 0.65x gateway and 1.8x worker to the
1x laptop. Idle node assignment minimizes modeled completion time;
priority uses watchlist relevance, corrections and aging. No inference
is skipped to manufacture a gain. Times are rounded to a 10 ms tick.
Remote attempts pay an assumed round trip and request/result byte sizes.
Worker outages only apply to the cooperative extra worker; source outages
apply to every architecture. The source buffer assumes eventual delivery,
unlike a real bounded RSS feed. Lost worker leases requeue after 200 ms;
late simulated responses are rejected, and duplicate packet IDs are ignored.
The replay includes the trace and per-attempt outcomes.

Metrics are virtual time to first/all relevant completion, p95 publication
to completion, makespan, count, retransmissions, stale-result rejection and
assumed network bytes. This is not a real hardware benchmark, a network
protocol, proof of exactly-once distributed execution, or a forecast of
market impact. The existing measured BGE benchmark remains distinct.
No inbox or configuration writes. API local-only, bounded and deterministic.

Run python -m unittest tests.test_edge_simulation -v. Full CI also checks
API behavior and JS syntax. Browser interaction should be verified separately.
