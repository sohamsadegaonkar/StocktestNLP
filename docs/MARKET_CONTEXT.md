# Disclosure market context (local integration contract)

GET /api/market-context/INFY?filing_date=2026-11-23

This endpoint is for the news-to-brief UI. It uses the same bounded, cached
Upstox daily-candle provider as the existing chart. It does not alter any NLP
model, training split, inbox record, or simulated replay behavior. Hosted mode
remains disabled for market-provider calls, consistent with the existing app.

Display the before/after daily closes and dates, price return percentage, Nifty
50 return percentage, and their difference in percentage points. Use the exact
same two sessions for both instruments. Show source timestamps, state, reason
and limitations. Null means unavailable: never substitute zero. The date is the
NSE disclosure date in IST; the filing day and current day are excluded, so the
comparison is conservative and may not be available for a new filing. No
interpolation, trading advice, causal attribution or price prediction is made.
A gap of more than seven calendar days on either side is not treated as a
surrounding-session comparison. Corporate action adjustments are not verified.
For older filings, the app requests a bounded daily range within seven calendar days on either side of the filing. Coverage depends on the provider. The client requests comparisons on demand for dated disclosures, including local imports.

For the UI owner: attach this response to the selected filing and only request
it for supported instrument symbols; do not map legacy TATAMOTORS to TMPV
without issuer/ISIN verification. A news card should still render its evidence
if the market endpoint fails. Keep this separate from the model's event labels
and qualitative monitoring checklist.

Validation: python -m unittest tests.test_market_context -v
Existing CI will run the full suite; fixtures are synthetic and do not establish
live provider coverage or causal accuracy.
