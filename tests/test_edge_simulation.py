import unittest

from stockbrief.edge_simulation import experiment


class EdgeSimulationTests(unittest.TestCase):
    def test_reproducibility_and_equal_workload(self):
        a = experiment()
        self.assertEqual(a, experiment())
        for run in a["runs"].values():
            self.assertEqual(run["completed"], 24)
            self.assertEqual(len({r["job"] for r in run["results"]}), 24)
            self.assertEqual(run["duplicates"], 4)
            self.assertEqual(set(r["job"] for r in run["results"]), set(range(1, 25)))

    def test_worker_loss_requeues_and_rejects_stale_result(self):
        run = experiment(scenario="worker_outage")["runs"]["cooperative"]
        self.assertGreater(run["lease_retries"], 0)
        self.assertEqual(run["lease_retries"], run["stale_results"])
        self.assertEqual(run["completed"], run["total"])
        for attempt in run["attempts"]:
            if attempt["node"] == "worker":
                self.assertFalse(200 <= attempt["start_ms"] < 1300)
        self.assertTrue(any(e["event"] == "lease_lost" for e in run["trace"]))
        self.assertTrue(any(e["event"] == "stale_result" for e in run["trace"]))

    def test_source_outage_defers_only_unreceived_work(self):
        runs = experiment(burst=False, scenario="source_outage")["runs"]
        for run in runs.values():
            for row in run["results"]:
                if 200 <= row["published_ms"] < 1200:
                    self.assertGreaterEqual(row["received_ms"], 1200)
                self.assertGreaterEqual(row["completed_ms"], row["received_ms"])
            self.assertTrue(any(200 <= r["completed_ms"] < 1200 for r in run["results"]))

    def test_high_network_delay_changes_allocation(self):
        low = experiment(latency=0, scenario="normal")["runs"]["cooperative"]
        high = experiment(latency=300, scenario="normal")["runs"]["cooperative"]
        self.assertNotEqual(low["makespan_ms"], high["makespan_ms"])
        self.assertEqual(high["completed"], high["total"])

    def test_adaptive_prioritizes_watchlist_without_changing_job_count(self):
        runs = experiment(scenario="normal")["runs"]
        self.assertLess(runs["adaptive"]["first_relevant_ms"], runs["fifo"]["first_relevant_ms"])
        self.assertEqual(runs["adaptive"]["total"], runs["fifo"]["total"])

    def test_boundaries(self):
        for count, delay in [(6,0),(60,300)]:
            for burst in (True,False):
                for scenario in ("normal","worker_outage","source_outage","combined"):
                    for run in experiment(count,burst,delay,scenario)["runs"].values():
                        self.assertEqual(run["completed"], count)
        for count, delay in [(0,30),(61,30),(12,-1),(12,301)]:
            with self.assertRaises(ValueError):
                experiment(count=count,latency=delay)
