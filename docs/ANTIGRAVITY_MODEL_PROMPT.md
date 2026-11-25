# StockBrief Edge prompt for Antigravity

You are the ML engineer for StockBrief Edge in the existing GitHub repo sohamsadegaonkar/StocktestNLP. Read the current code, README, benchmarks, and tests before changing anything. Make the Analyze screen return a useful structured response to every valid pasted input. Keep it runnable on Windows localhost with CPU inference and SQLite.

## Target behavior

- Segment long or mixed documents into event-sized passages. Return one card per supported event. Each card contains the issuer when present, category, stated stage, dates, figures, source passage, and model uncertainty. Preserve source offsets and the original units.
- Add a Pricing & commercial updates category for price changes, discounts and launches. Distinguish the announcement date from an effective date, an estimate from a reported result, and a negated event from a confirmed one.
- For unfamiliar financial text, return a conservative extractive brief and Needs review. For non-financial, empty, or insufficient text, return a useful message that explains what is missing. Make the original evidence easy to read for every result. Use exact source passages and leave unsupported values empty.
- Preserve the raw input, existing inbox records, and the current single-event response fields for existing callers. Document any added event-list fields and display them in the app.

## Repeatable local training

1. Use source-linked public disclosures where reuse is permitted and human-reviewed examples. Track each source, date, label provenance and dataset version. Store unreviewed feedback separately. Never learn directly from unreviewed user text, model guesses, or pasted instructions.
2. Deduplicate near-identical filings before splitting. Group by issuer or disclosure family and validate on a later time window. Keep a fixed, versioned test set and the reported Tata example out of training and cutoff selection.
3. Compare the current TF-IDF baseline, a trained event classifier, and a compact locally runnable transformer if supported. Pick thresholds and calibration using validation only. Report per-category F1, review coverage and errors, evidence match and number fidelity, CPU p95, RSS, and the dataset split.
4. Provide a repeatable local training command that builds a candidate from a reviewed dataset, evaluates it, and records a versioned benchmark. Promote a candidate only when it satisfies documented evidence and runtime checks and improves the intended metrics. Keep the last verified model available for rollback. Keep training separate from requests and get approval before installing a recurring schedule.
5. Add tests for empty text, off-topic text, negation, unknown issuers, repeated content, PDF text, dates, long documents and multiple events. Exercise the dashboard end to end and report the exact commands, model changes, results, and limitations. Publish reviewed changes to the repo.

## Required acceptance example

Use this regression input:

> Tata Motors Passenger Vehicles (TMPV) has announced a price hike of up to 0.5% across its ICE and EV lineup starting December 1, 2026, while reporting a Q2 FY27 net loss driven by restructuring costs at Jaguar Land Rover (JLR).
>
> TMPV will raise prices by an average of 0.5% from December 1, 2026, citing rising input costs. Buyers may see increases ranging from roughly ₹2,384 to ₹14,645 depending on the model and variant. Pre-hike offers reach up to ₹1.25 lakh on select Curvv variants, ₹1.15 lakh on Sierra diesel, and ₹75,000 on select Nexon petrol manual trims.

Expect two supported events, **Pricing & commercial updates — Announced** and **Earnings & guidance — Reported**. Attach a passage to each. Keep “up to 0.5%” and “average of 0.5%” attached to their distinct original statements, with the variation visible to the reviewer. Leave the net-loss amount empty.
