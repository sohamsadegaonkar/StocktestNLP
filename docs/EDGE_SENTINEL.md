# Trader Edge Sentinel — milestone 1

The local NSE reader now commits received RSS rows to SQLite before running
inference. The queue survives process restart and source disconnection. Each
completion commits the disclosure, receipt and processing history in one
transaction. The queue is bounded at 200 rows; capacity-deferred rows may be
lost if the RSS window rolls over. Failed jobs are retained after three
validation failures; operator recovery is a future task. The latest 200
completion records are retained; receipts remain persistent.

Choose a local watchlist and FIFO or adaptive policy in Edge Lab. The adaptive
policy gives a bounded relevance/correction/urgency boost with unbounded aging.
This is deterministic lightweight triage, not a trading signal or learned
urgency model. Host CPU >=80% or available RAM <512 MiB selects one job rather
than five; this is a batch-size heuristic, not a hard CPU/RAM limit. A running
inference is nonpreemptive. Offline drain uses already received RSS only and
makes no provider call, even when permitted PDF retrieval is enabled.
RSS/PDF scope remains explicit. Normal live refresh still attempts permitted
PDF retrieval if enabled. No scheduler runs when the application is closed.

Same issuer and exact nonempty attachment URL permit a comparison with the last
processed RSS excerpt. This is a candidate relationship, not a proven official
revision or chronological correction. Changed figures and bounded text diffs
are reported; changed data is never discarded solely for semantic similarity.
When both analyzed records have matching model versions and source scopes,
compare their inferred category/stage sets as an additional model-level delta.
No cross-link correlation or automatic claim of materiality yet.

Compare FIFO and adaptive using six identical synthetic documents, actual
local model inference and two alternating passes each after warmup. Export
measured completion times and category-equivalence check. No synthetic
latency, energy estimates or cloud speed claims. Benchmark is synchronous,
local-only, isolated from live refresh; it does not create inbox records.
The legacy simulated link/display remain secondary. No broker integration,
network listener changes, peer authentication, real second-device execution
or new training. Cooperative execution is simulated only by user decision.
See [cooperative simulation](EDGE_SIMULATION.md).

Run `python -m unittest tests.test_edge tests.test_live_feed -v` and full CI.
Actual model benchmark available on the Edge Lab screen after startup.
