# Local live NSE announcements

NSE publishes an official RSS subscription at https://www.nseindia.in/static/rss-feed.
The announcement feed URL is https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml.
The exchange restricts automated website collection; this implementation uses
the advertised RSS subscription, not an undocumented website endpoint.
See https://betanseapi.nseindia.com/static/nse-terms-of-use and verify usage
rights before expanding beyond RSS. No paid subscription or account is created.

Start the local server normally, open Inbox, and click Start auto refresh or
Fetch now. The browser polls every two minutes while Inbox is visible and
auto refresh is enabled. It stops with the tab or toggle; no external scheduled
job is installed. Hosted mode cannot refresh the feed. The RSS parser matches
five known issuer names exactly; other issuers, invalid times, and unsupported
attachments are not guessed. The latest 50 matching feed rows are considered,
five new records per poll. This is a bounded watchlist reader, not an exhaustive
exchange archive. The feed can be delayed and may roll over.

RSS excerpts are the default. Only where automated retrieval and analysis of
the linked filing are permitted, start the server with
STOCKBRIEF_NSE_ATTACHMENTS=1 to retrieve the allowlisted archive PDF. It is
limited to 5 MB, no redirects, 30 text pages and existing text limits. A
failed/scanned/oversized PDF uses the RSS excerpt, clearly marked; stored
excerpt records do not automatically upgrade later. No source is used for
training. Publisher timestamps, retrieval timestamps and evidence scope are
saved. Deduplication receipts survive restart/reset in the same SQLite
transaction as the analysis. Corrected descriptions can create new entries.

Each new record passes through the existing local model and intelligence
endpoint. Selecting it shows a matched company/Nifty 50 daily return
comparison if complete surrounding sessions are available; the issuer chart
can be selected from that panel. New filings normally lack later completed
sessions, so pending is expected. No causal or predictive price claim.

Tests use synthetic RSS, mocked retrieval, and the existing model in normal CI.
Run python -m unittest tests.test_live_feed -v. Live XML and PDF retrieval
were not verified in the chat runtime (DNS unavailable). Verify feed schema
and publisher timestamp behavior on the target localhost network before
claiming real-data end-to-end success.
