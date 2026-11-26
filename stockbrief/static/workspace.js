const $ = (selector) => document.querySelector(selector);
const viewText = {
  inbox: ["Disclosure workspace", "Find the event, its stage, and the source passage behind each company announcement."],
  analyze: ["Analyze a disclosure", "Classify a company filing and inspect the passages used in the result."],
  lab: ["Edge lab", "Measure local financial inference, adaptive scheduling and persistent event recovery."],
  evaluation: ["Evaluation", "Compare event classification, review decisions, and inference latency."],
};
const originNames = {
  historical: "Historical paraphrase",
  synthetic: "Synthetic scenario",
  "nse-rss": "NSE RSS excerpt",
  "nse-pdf": "NSE filing PDF text",
  text: "Text import",
  pdf: "PDF import",
};
let state;
let selection;
let replayBusy = false;
let analysisBusy = false;
let workspaceBusy = false;
let displayRevision = 0;
let detailEntry;

const inboxControls = ["filter-search", "filter-ticker", "filter-category"];
const inboxStorageKey = "stockbrief-inbox-view";

function restoreInboxView() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(inboxStorageKey));
    if (!saved) return;
    selection = saved.selection;
    inboxControls.forEach((id) => {
      const input = $("#" + id);
      input.value = typeof saved[id] === "string" ? saved[id] : "";
      if (input.tagName === "SELECT" && input.selectedIndex < 0) input.selectedIndex = 0;
    });
  } catch {
    // Session storage is optional in private or restricted browser contexts.
  }
}

function saveInboxView() {
  const values = Object.fromEntries(inboxControls.map((id) => [id, $("#" + id).value]));
  try {
    sessionStorage.setItem(inboxStorageKey, JSON.stringify({...values, selection}));
  } catch {
    // Inbox navigation remains usable without session storage.
  }
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function dateLabel(raw) {
  if (!raw) return "";
  const date = new Date(raw);
  return Number.isNaN(date.getTime()) ? raw : new Intl.DateTimeFormat("en-IN", {dateStyle: "medium", timeZone: "UTC"}).format(date);
}

function percent(value) {
  return value == null ? "—" : (100 * value).toFixed(1) + "%";
}

function status(target, message, isError = false) {
  target.textContent = message;
  target.classList.toggle("error", isError);
}

function globalError(error) {
  $("#global-error").hidden = !error;
  $("#global-error-text").textContent = error?.message || "";
}

async function api(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("Could not connect to the workspace. Check your connection and try again.");
  }
  if (!response.ok) {
    const fallback = "Request failed (" + response.status + ").";
    let message = fallback;
    if (response.headers.get("content-type")?.includes("application/json")) {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } else {
      message = (await response.text()) || fallback;
    }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

function presentView() {
  const key = location.hash.slice(1);
  if (key === "main") return;
  const current = Object.hasOwn(viewText, key) ? key : "inbox";
  for (const id of Object.keys(viewText)) $("#view-" + id).hidden = id !== current;
  for (const link of document.querySelectorAll("nav [data-view]")) {
    if (link.dataset.view === (key === "market-panel" ? key : current)) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  $("#view-title").textContent = viewText[current][0];
  $("#view-description").textContent = viewText[current][1];
  $(".page-head").dataset.current = current;
  $("#hero-actions").hidden = current !== "inbox";
  $("#hero-process").hidden = current !== "inbox";
  document.title = viewText[current][0] + " · StockBrief Edge";
  if (current !== "lab") stopSimulation();
  if (current === "lab" && state?.live_feed_available) loadEdge();
  if (key === "market-panel") requestAnimationFrame(() => {
    $("#market-heading").focus({preventScroll: true});
    $("#market-panel").scrollIntoView({block: "start"});
  });
}

function addOptions(select, entries) {
  for (const [value, name] of Object.entries(entries)) {
    const option = element("option", name);
    option.value = value;
    select.append(option);
  }
}

function sourceLink(url, text) {
  const link = element("a", text, "source-link");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function detailField(title, value) {
  const pair = element("div");
  pair.append(element("dt", title), element("dd", value));
  return pair;
}

function downloadJson(name, payload) {
  const file = new Blob([JSON.stringify(payload, null, 2) + "\n"], {type: "application/json"});
  const url = URL.createObjectURL(file);
  const link = element("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function eventDetails(entry, brief, pane, separateCard) {
  const facts = element("dl");
  if (!separateCard) facts.append(detailField("Event category", brief.category_name));
  facts.append(detailField("Disclosure stage", brief.stage));
  pane.append(facts);

  const evidenceList = element("div", undefined, "evidence-list");
  evidenceList.append(element("h3", "Source evidence", "eyebrow"));
  for (const evidence of brief.evidence) {
    const quote = element("blockquote", undefined, "evidence");
    const location = entry.origin === "historical" ? "Fixture excerpt " : "Page ";
    const isStage = brief.stage_evidence?.page === evidence.page && brief.stage_evidence?.start === evidence.start;
    quote.append(element("strong", location + evidence.page + (isStage ? " · Stage evidence" : "")), element("p", evidence.text));
    evidenceList.append(quote);
  }
  pane.append(evidenceList);

  for (const [label, values] of [["Figures in the source text", brief.figures], ["Dates and periods", brief.dates]]) {
    if (!values?.length) continue;
    const figures = element("div", undefined, "amounts");
    const chips = element("div", undefined, "chips");
    const figuresByValue = new Map(values.map((figure) => [figure.value, figure]));
    for (const figure of figuresByValue.values()) {
      const locator = entry.origin === "historical" ? "Excerpt " : "Page ";
      chips.append(element("span", figure.value + " · " + locator + figure.page, "chip subtle"));
    }
    figures.append(element("h3", label, "eyebrow"), chips);
    pane.append(figures);
  }
}

function signedPct(value) {
  if (value == null) return "Unavailable";
  return (value > 0 ? "+" : "") + Number(value).toFixed(2) + "%";
}

function horizonSummaryCard(title, stats) {
  const card = element("div", undefined, "historical-horizon");
  card.append(element("h4", title + " · " + stats.completed_count + " completed" +
    (stats.pending_after_cutoff_count ? " (" + stats.pending_after_cutoff_count + " excluded at cutoff)" : "")));
  if (!stats.completed_count) {
    card.append(element("p", "No matched window finished before the cutoff date.", "small muted"));
    return card;
  }
  const metrics = element("dl");
  metrics.append(
    detailField("Raw return (median)", signedPct(stats.raw_return_median_pct)),
    detailField("Excess vs Nifty 50 (median)", signedPct(stats.excess_nifty50_median_pct) +
      (stats.excess_nifty50_iqr_pct ? " · IQR " + signedPct(stats.excess_nifty50_iqr_pct[0]) + " to " + signedPct(stats.excess_nifty50_iqr_pct[1]) : "")),
    detailField("Market-model CAR (median)", signedPct(stats.market_model_car_median_pct)),
    detailField("Positive Nifty excess", stats.positive_excess_count + " of " + stats.completed_count),
  );
  card.append(metrics);
  return card;
}

function windowMetricsCell(observed, statusFlag, endDate, cutoffDate) {
  const box = element("div", undefined, "comp-metrics");
  if (!observed) {
    box.append(element("span", "Excluded (" + (endDate || statusFlag) + " >= " + cutoffDate + ")", "chip review"));
    return box;
  }
  box.append(
    element("span", "Raw: " + signedPct(observed.raw_return_pct), "small"),
    element("span", "vs Nifty 50: " + signedPct(observed.excess_vs_nifty50_pct), "small"),
    element("span", "CAR: " + signedPct(observed.market_model_car_pct), "small"),
    element("span", "Ended " + dateLabel(endDate), "small muted"),
  );
  return box;
}

function historicalComparablesPanel(entry) {
  const panel = element("section", undefined, "intelligence-panel historical-panel");
  const head = element("div", undefined, "actions");
  head.append(element("h3", "Similar past events"), element("span", "Historical analytics only · no predictions", "chip subtle"));
  const body = element("div");
  panel.append(head, body);

  async function load() {
    const wait = element("p", "Searching strictly prior historical NSE filings…", "small muted");
    wait.setAttribute("role", "status");
    body.replaceChildren(wait);
    try {
      const result = await api("/api/disclosures/" + entry.id + "/comparables");
      body.replaceChildren();
      const meta = element("p",
        result.package_label + " · " +
        (result.matching_mode === "semantic_and_category" ? "Semantic + category similarity" :
          result.matching_mode === "category_fallback" ? "Category matching fallback" : "No matching applied") +
        " · Cutoff " + timeInIndia(result.cutoff_ist),
        "small muted");
      body.append(meta, element("p", result.cutoff_assumption, "small muted"));

      if (result.state === "disabled" || result.state === "unavailable" || result.state === "insufficient_evidence") {
        const reasonText = result.limitations?.[0] || "Historical comparison is unavailable for this disclosure.";
        body.append(element("p", reasonText, "small"));
      } else {
        const dist = element("div", undefined, "historical-summary");
        dist.append(
          horizonSummaryCard("1-day observed window (T0)", result.summary_distribution.horizon_1d),
          horizonSummaryCard("5-day observed window (T0..T0+4)", result.summary_distribution.horizon_5d),
        );
        body.append(dist, table("Comparable past NSE disclosures (strictly before cutoff)",
          ["Prior filing & match basis", "1-day observed", "5-day observed"],
          result.comparables.map((comp) => {
            const left = element("div", undefined, "comp-filing");
            const badges = element("div", undefined, "chips");
            badges.append(
              element("span", comp.symbol, "chip"),
              element("span", comp.category_name + " · " + comp.stage, "chip subtle"),
            );
            if (!comp.corp_action_verified) badges.append(element("span", "Unadjusted fallback", "chip review"));
            if (comp.has_split_near || comp.has_div_near) badges.append(element("span", "Near split/div", "chip review"));
            if (comp.is_overlapping_10d) badges.append(element("span", "Overlapping <=10d", "chip subtle"));
            left.append(
              badges,
              element("p", dateLabel(comp.published_ts_ist.slice(0, 10)) + " (" + comp.published_ts_ist + ") · " + comp.sector, "small muted"),
              element("p", comp.excerpt, "comp-excerpt"),
              element("p", comp.match_reasons.join(" · "), "small muted"),
            );
            if (comp.source_url) left.append(sourceLink(comp.source_url, "NSE source →"));
            return [
              left,
              windowMetricsCell(comp.observed_1d, comp.window_1d_status, comp.window_1d_end_date, result.cutoff_date),
              windowMetricsCell(comp.observed_5d, comp.window_5d_status, comp.window_5d_end_date, result.cutoff_date),
            ];
          })));
      }

      const limits = element("details");
      const list = element("ul");
      (result.limitations || []).forEach((item) => list.append(element("li", item)));
      limits.append(element("summary", "Method, cutoffs & limitations"), list);
      body.append(limits);
      if (result.comparables?.length) {
        const download = element("button", "Download historical comparisons", "button");
        download.type = "button";
        download.addEventListener("click", () => downloadJson("stockbrief-comparables-" + entry.id + ".json", result));
        body.append(download);
      }
    } catch (error) {
      status(wait, error.message, true);
      const retry = element("button", "Retry historical lookup", "button");
      retry.type = "button";
      retry.addEventListener("click", load);
      body.append(retry);
    }
  }
  load();
  return panel;
}

function intelligencePanel(entry) {
  const panel = element("section", undefined, "intelligence-panel");
  panel.append(element("h3", "News intelligence"));
  const body = element("div");
  panel.append(body);
  async function load() {
    const message = element("p", "Building evidence-backed brief…", "small muted");
    message.setAttribute("role", "status");
    body.replaceChildren(message);
    try {
      const result = await api("/api/disclosures/" + entry.id + "/intelligence");
      body.replaceChildren(element("p", result.state === "needs_review"
        ? "Needs review — no supported event established."
        : "Extractive brief · model interpretation", "small muted"));
      for (const event of result.events) {
        const section = element("article", undefined, "intelligence-event");
        section.append(element("h4", event.category_name + " · " + event.stage));
        if (event.evidence.length) {
          const quote = element("blockquote", undefined, "evidence");
          const item = event.evidence[0];
          quote.append(element("strong", "What the stored source says · " +
            (entry.origin === "historical" ? "Excerpt " : "Page ") + item.page),
            element("p", item.text));
          section.append(quote);
        } else {
          section.append(element("p", "No validated evidence span available.", "small muted"));
        }
        if (event.missing_guidance) {
          section.append(element("h4", "What is missing & how to continue"), element("p", event.missing_guidance));
        }
        if (event.interpretation) {
          section.append(element("h4", "Why it may matter"), element("p", event.interpretation));
        }
        if (event.potentially_affected) {
          section.append(element("h4", "Potentially affected"), element("p", event.potentially_affected));
        }
        const list = element("ul");
        event.monitor.forEach(item => list.append(element("li", item)));
        section.append(element("h4", "What to verify next"), list);
        body.append(section);
      }
      const limits = element("details");
      const notes = element("ul");
      result.limitations.forEach(item => notes.append(element("li", item)));
      limits.append(element("summary", "Method and limitations"), notes);
      const download = element("button", "Download intelligence brief", "button");
      download.type = "button";
      download.addEventListener("click", () => downloadJson("stockbrief-intelligence-" + entry.id + ".json", result));
      body.append(limits, download);
    } catch (error) {
      status(message, error.message, true);
      const retry = element("button", "Retry intelligence brief", "button");
      retry.type = "button";
      retry.addEventListener("click", load);
      body.append(retry);
    }
  }
  load();
  return panel;
}

function marketContextPanel(entry) {
  const panel = element("section", undefined, "intelligence-panel");
  const actions = element("div", undefined, "actions");
  const compare = element("button", "Compare market closes", "button");
  compare.type = "button";
  const chartLink = element("a", "Show issuer price chart", "small");
  chartLink.href = "#market-panel";
  chartLink.addEventListener("click", () => {
    const chart = $("#market-symbol");
    chart.value = entry.ticker;
    chart.dispatchEvent(new Event("change"));
  });
  actions.append(compare, chartLink);
  panel.append(element("h3", "Observed market context"), actions);
  const note = element("p", "", "small muted");
  note.setAttribute("role", "status");
  const output = element("div");
  panel.append(note, output);
  const signed = (n, unit) => n == null ? "Unavailable" : (n > 0 ? "+" : "") + n.toFixed(2) + unit;
  const reasons = {
    future_filing_date: "This disclosure date is in the future.",
    symbol_data_unavailable: "Upstox returned no completed prices for this issuer and date.",
    no_prior_completed_session: "The prior completed trading session is missing.",
    no_later_completed_session: "A later completed trading session is unavailable for comparison.",
    session_gap_too_large: "The surrounding closes are too far from this disclosure date.",
  };
  compare.addEventListener("click", async () => {
    compare.disabled = true;
    output.replaceChildren();
    status(note, "Checking completed trading sessions…");
    try {
      const result = await api("/api/market-context/" + encodeURIComponent(entry.ticker) + "?filing_date=" + encodeURIComponent(entry.date));
      if (!result.before || !result.after) {
        status(note, reasons[result.reason] || "No matching trading sessions are available.");
        return;
      }
      const metrics = element("dl");
      metrics.append(detailField("Company return", signed(result.symbol_return_pct, "%")),
        detailField("Nifty 50 return", signed(result.benchmark_return_pct, "%")),
        detailField("Relative return", signed(result.relative_return_pp, " pp")));
      status(note, dateLabel(result.before.date) + " to " + dateLabel(result.after.date) +
        (result.state === "cached" ? " · Cached prices" : ""));
      const list = element("ul");
      result.limitations.forEach(item => list.append(element("li", item)));
      const checked = result.symbol_checked_at ? " · Checked " + timeInIndia(result.symbol_checked_at) : "";
      output.append(metrics, element("p", "Closes ₹" + result.before.close + " → ₹" + result.after.close + " · Upstox" + checked, "small muted"), list);
    } catch (error) {
      status(note, error.message, true);
    } finally {
      compare.disabled = false;
    }
  });
  return panel;
}

function detail(entry, pane) {
  const brief = entry.brief;
  const heading = element("div", undefined, "chips");
  heading.append(element("span", entry.ticker, "chip"), element("span", originNames[entry.origin] || "Local import", "chip subtle"));
  if (brief.category === "review") heading.append(element("span", "Needs review", "chip review"));
  const meta = [entry.source, dateLabel(entry.date)].filter(Boolean).join(" · ");
  pane.replaceChildren(heading, element("h2", entry.title), element("p", meta, "source-meta"));
  if (brief.category === "review") {
    const note = brief.review_message ?? ("Leading candidate: " + brief.candidate_name + ". Review the source excerpts.");
    pane.append(element("p", note, "small muted"));
  }
  if (brief.chunk_limit_reached) {
    pane.append(element("p", "The analysis reached its 240-chunk limit. Later text may need a separate review.", "small muted"));
  }
  if (entry.published_at) pane.append(element("p", "Published " + entry.published_at +
    " · retrieved " + entry.retrieved_at, "small muted"));
  if (entry.source_scope === "rss_excerpt") pane.append(element("p",
    "RSS excerpt only — the full filing was not analyzed.", "small muted"));
  pane.append(intelligencePanel(entry));
  if (brief.events?.length > 1) {
    for (const event of brief.events) {
      const card = element("section", undefined, "event-card");
      card.append(element("h3", event.category_name));
      eventDetails(entry, event, card, true);
      pane.append(card);
    }
  } else {
    eventDetails(entry, brief, pane, false);
  }

  if (entry.date && entry.origin !== "synthetic" && state?.market_tickers?.includes(entry.ticker)) {
    pane.append(marketContextPanel(entry));
  }
  if (state?.historical_available !== false) {
    pane.append(historicalComparablesPanel(entry));
  }

  const sourceActions = element("div", undefined, "actions source-actions");
  if (entry.url) sourceActions.append(sourceLink(entry.url, "Open original source →"));
  const download = element("button", "Download brief", "button");
  download.type = "button";
  download.addEventListener("click", () => downloadJson("stockbrief-" + entry.id + ".json", entry));
  sourceActions.append(download);
  pane.append(sourceActions);
  const source = element("details");
  source.append(element("summary", entry.origin === "historical" ? "Read historical fixture text" : "Read extracted document text"));
  source.addEventListener("toggle", async () => {
    if (!source.open || source.children.length > 1) return;
    const wait = element("p", "Loading text…", "small");
    source.append(wait);
    try {
      const record = await api("/api/disclosures/" + entry.id);
      wait.remove();
      record.pages.forEach((page, index) => {
        const label = entry.origin === "historical" ? "Excerpt " : "Page ";
        source.append(element("p", label + (index + 1), "eyebrow"), element("pre", page || "No selectable text."));
      });
    } catch (error) {
      status(wait, error.message, true);
      source.addEventListener("toggle", () => { if (!source.open) wait.remove(); }, {once: true});
    }
  });
  pane.append(source);
  const model = element("details");
  const scores = element("p", brief.model + " · " + brief.chunk_count + " chunks · " + brief.inference_ms + " ms on host", "small muted");
  const best = element("p", "Top model score " + percent(brief.score) + ". Scores rank candidate categories.", "small muted");
  model.append(element("summary", "Model details"), scores, best);
  pane.append(model);
  if (state?.transformer) {
    const run = element("button", "Run transformer second opinion", "button");
    const result = element("p", "", "form-status");
    result.setAttribute("role", "status");
    run.addEventListener("click", async () => {
      run.disabled = true;
      status(result, "Running the transformer…");
      try {
        const prediction = await api("/api/disclosures/" + entry.id + "/second-opinion");
        status(result, prediction.name + " · " + prediction.model + " · First six chunks");
      } catch (error) {
        status(result, error.message, true);
      } finally {
        run.disabled = false;
      }
    });
    pane.append(run, result);
  }
}

function renderInbox(focusId) {
  if (!state) return;
  const ticker = $("#filter-ticker").value;
  const category = $("#filter-category").value;
  const query = $("#filter-search").value.trim().toLowerCase();
  const entries = state.disclosures.filter((entry) => {
    const events = entry.brief.events?.length ? entry.brief.events : [entry.brief];
    const searchable = [entry.ticker, state.tickers[entry.ticker], entry.title, entry.source,
      ...events.flatMap((event) => [event.category_name, event.stage])];
    return (!ticker || entry.ticker === ticker) && (!category || events.some((event) => event.category === category)) &&
      (!query || searchable.some((value) => value?.toLowerCase().includes(query)));
  });
  $("#filter-count").textContent = entries.length + " of " + state.disclosures.length;
  if (!entries.some((entry) => entry.id === selection)) selection = entries[0]?.id;
  const rows = $("#disclosure-list");
  const pane = $("#disclosure-detail");
  rows.replaceChildren();
  for (const entry of entries) {
    const row = element("button", undefined, "document-row");
    row.type = "button";
    row.dataset.id = entry.id;
    row.setAttribute("aria-pressed", String(entry.id === selection));
    const meta = element("span", undefined, "row-meta");
    meta.append(element("span", entry.ticker, "row-ticker"), element("span", dateLabel(entry.date) || "Undated", "row-date"));
    const event = entry.brief.events?.length > 1
      ? entry.brief.events.map((item) => item.category_name).join(" · ")
      : entry.brief.category_name + (entry.brief.stage === "Unclear" ? "" : " · " + entry.brief.stage);
    row.append(meta, element("strong", entry.title), element("span", event, "row-event"));
    row.addEventListener("click", () => {
      selection = entry.id;
      renderInbox(entry.id);
      if (matchMedia("(max-width: 760px)").matches) pane.focus();
    });
    const item = element("li");
    item.append(row);
    rows.append(item);
  }
  if (focusId) rows.querySelector('[data-id="' + focusId + '"]')?.focus();
  const selected = entries.find((entry) => entry.id === selection);
  if (selected) {
    if (detailEntry !== selected) detail(selected, pane);
  } else {
    const empty = element("li", "No disclosures match these filters.", "list-state");
    const clear = element("button", "Clear filters", "button");
    clear.type = "button";
    clear.addEventListener("click", () => {
      inboxControls.forEach((id) => { $("#" + id).value = ""; });
      renderInbox();
      $("#filter-search").focus();
    });
    empty.append(clear);
    rows.append(empty);
    pane.replaceChildren(element("h2", "No disclosure selected"), element("p", "Change the filters to review a disclosure and its source evidence.", "muted"));
  }
  detailEntry = selected;
  saveInboxView();
}

let liveEnabled = false;
let liveBusy = false;
let liveStarted = 0;
const indiaTime = new Intl.DateTimeFormat("en-IN", {timeZone: "Asia/Kolkata", dateStyle: "medium", timeStyle: "short"});
const timeInIndia = (stamp) => {
  if (!stamp) return "—";
  const parsed = new Date(stamp);
  return Number.isNaN(parsed.getTime()) ? String(stamp) : indiaTime.format(parsed) + " IST";
};
async function pollLive(showAlternatives = false) {
  if (!state?.live_feed_available || liveBusy || analysisBusy || replayBusy || workspaceBusy) return;
  liveBusy = true;
  liveStarted = Date.now();
  const button = $("#live-refresh");
  const note = $("#live-status");
  button.disabled = true;
  status(note, "Checking NSE RSS…");
  try {
    const result = await api("/api/live/refresh", {method: "POST"});
    if (result.state === "busy") {
      status(note, "Another refresh is running.");
      return;
    }
    const summary = "Imported " + result.imported + " · " + result.duplicates + " already seen · " +
      result.pending + " pending in local queue · " + result.errors + " processing warnings" +
      (result.capacity_deferred ? " · " + result.capacity_deferred + " deferred at capacity" : "");
    const lead = {
      available: result.matching + " matches in the current NSE snapshot. ",
      unavailable: "Could not reach NSE RSS. Saved briefs are available below. ",
      no_matches: "NSE RSS responded. This snapshot has no filings for the five tracked companies. ",
    };
    status(note, (lead[result.state] || "") + summary +
      " · Last check: " + (result.checked_at ? timeInIndia(result.checked_at) : "none") +
      " · Last successful check: " + (result.last_success ? timeInIndia(result.last_success) : "none") +
      " · Latest publication time: " + (result.latest_supported_at ? timeInIndia(result.latest_supported_at) : "none in snapshot") +
      (result.throttled ? " · Cached refresh status" : ""), result.state === "unavailable");
    if (showAlternatives && ["no_matches", "unavailable"].includes(result.state)) $("#headline-panel").open = true;
    if (result.imported || result.throttled) await refresh();
  } catch (error) {
    status(note, error.message, true);
  } finally {
    liveBusy = false;
    button.disabled = false;
  }
}
$("#live-refresh").addEventListener("click", () => pollLive(true));
$("#live-toggle").addEventListener("click", () => {
  liveEnabled = !liveEnabled;
  $("#live-toggle").textContent = liveEnabled ? "Stop auto refresh" : "Start auto refresh";
  if (liveEnabled) pollLive();
});
setInterval(() => {
  if (liveEnabled && !document.hidden && !$("#view-inbox").hidden && Date.now() - liveStarted >= 120000) pollLive();
}, 5000);

let headlineRequest = 0;
$("#headline-symbol").addEventListener("change", () => {
  ++headlineRequest;
  $("#headline-list").replaceChildren();
  status($("#headline-status"), "");
  $("#headline-load").disabled = false;
});
$("#headline-load").addEventListener("click", async () => {
  const request = ++headlineRequest;
  const button = $("#headline-load");
  const note = $("#headline-status");
  button.disabled = true;
  status(note, "Looking for recent headlines…");
  try {
    const result = await api("/api/headlines/" + encodeURIComponent($("#headline-symbol").value));
    if (request !== headlineRequest) return;
    const messages = {
      available: "Checked " + (result.checked_at ? timeInIndia(result.checked_at) : "just now"),
      cached: "Showing cached headlines from " + (result.checked_at ? timeInIndia(result.checked_at) : "the last check"),
      empty: "No matching headlines were found within the past seven days.",
      unavailable: "The headline feed is unavailable. You can still paste a source in Analyze.",
    };
    status(note, messages[result.state] || "Headline feed unavailable.", result.state === "unavailable");
    $("#headline-list").replaceChildren(...result.items.map(item => {
      const row = element("li");
      row.append(sourceLink(item.url, item.title), element("p", item.publisher + " · " + timeInIndia(item.published_at), "small muted"));
      return row;
    }));
  } catch (error) {
    if (request === headlineRequest) status(note, error.message, true);
  } finally {
    if (request === headlineRequest) button.disabled = false;
  }
});

let simReport;
let simTimer;
function stopSimulation() {
  if (simTimer !== undefined) clearInterval(simTimer);
  simTimer = undefined;
  $("#sim-play").textContent = "Play";
}
function simFrame() {
  if (!simReport) return;
  const run = simReport.runs[$("#sim-architecture").value];
  const t = Number($("#sim-clock").value);
  const down = window => window && window[0] <= t && t < window[1];
  const completed = run.results.filter(r => r.completed_ms <= t).length;
  const events = run.trace.filter(e => e.time_ms <= t);
  const received = events.filter(e => e.event === "received").length;
  $("#sim-time").textContent = t + " ms virtual time · " + completed + "/" + run.total + " completed · " +
    received + " received · source " + (down(run.source_outage) ? "offline" : "online (simulated)");
  const nodes = $("#sim-nodes"); nodes.replaceChildren();
  for (const node of run.nodes) {
    const card = element("article");
    const active = run.attempts.find(a => a.node === node.name && a.start_ms <= t && (a.end_ms === null || t < a.end_ms));
    const count = run.results.filter(r => r.node === node.name && r.completed_ms <= t).length;
    const state = node.name === "worker" && down(run.worker_outage) ? "Offline" : active ? "Job " + active.job : "Idle";
    card.append(element("p", node.name + " · " + node.speed + "×", "eyebrow"),
      element("strong", state), element("span", count + " completed"));
    nodes.append(card);
  }
  const panel = $("#sim-events");
  panel.replaceChildren(element("h3", "Latest simulated events"));
  if (events.length) panel.append(table("Virtual event trace", ["Time (ms)", "Event", "Job", "Device"],
    events.slice(-8).reverse().map(e => [String(e.time_ms), e.event.replaceAll("_"," "), String(e.job ?? "—"), e.node || "—"])));
}
function simSelect() {
  stopSimulation();
  if (!simReport) return;
  $("#sim-clock").max = simReport.runs[$("#sim-architecture").value].end_ms;
  $("#sim-clock").value = 0;
  simFrame();
}
$("#sim-architecture").addEventListener("change", simSelect);
$("#sim-clock").addEventListener("input", () => { stopSimulation(); simFrame(); });
$("#sim-reset").addEventListener("click", simSelect);
$("#sim-play").addEventListener("click", () => {
  if (simTimer !== undefined) { stopSimulation(); return; }
  if (!simReport) return;
  const clock = $("#sim-clock");
  if (Number(clock.value) >= Number(clock.max)) clock.value = 0;
  $("#sim-play").textContent = "Pause";
  simTimer = setInterval(() => {
    clock.value = Math.min(Number(clock.max), Number(clock.value) + Math.max(10, Math.ceil(Number(clock.max)/100/10)*10));
    simFrame();
    if (Number(clock.value) >= Number(clock.max)) stopSimulation();
  }, 100);
});
$("#sim-config").addEventListener("submit", async event => {
  event.preventDefault(); stopSimulation();
  const button = $("#sim-run");
  button.disabled = true;
  status($("#sim-status"), "Running virtual workload…");
  try {
    const data = await api("/api/edge/simulate", {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({count:Number($("#sim-count").value), latency_ms:Number($("#sim-delay").value),
        burst:$("#sim-burst").value === "burst",scenario:$("#sim-scenario").value})});
    simReport = data;
    const target = $("#sim-results");
    const runs = Object.values(data.runs);
    const ms = n => n === null ? "Unavailable" : n + " ms";
    target.replaceChildren(element("h3","Simulated architecture comparison"), element("p",data.scope,"small muted"),
      table("Virtual performance — same work", ["Architecture","First relevant","All relevant","p95 latency","Makespan","Completed"],
        runs.map(r => [r.architecture,ms(r.first_relevant_ms),ms(r.all_relevant_ms),ms(r.p95_latency_ms),ms(r.makespan_ms),r.completed+"/"+r.total])),
      table("Virtual reliability and network costs", ["Architecture","Duplicate packets","Lease retries","Stale results rejected","Assumed bytes"],
        runs.map(r => [r.architecture,r.duplicates,r.lease_retries,r.stale_results,r.modeled_network_bytes])));
    const bars = element("div", undefined, "sim-bars");
    bars.append(element("h3", "Virtual time to first relevant brief"));
    const maximum = Math.max(1,...runs.map(r => r.first_relevant_ms || 0));
    for (const r of runs) {
      const label = element("label",r.architecture+" · "+ms(r.first_relevant_ms));
      const meter = element("meter"); meter.min=0; meter.max=maximum; meter.value=r.first_relevant_ms ?? maximum;
      meter.setAttribute("aria-label",label.textContent); label.append(meter); bars.append(label);
    }
    const download = element("button","Export complete simulation trace","button");
    download.type="button"; download.addEventListener("click",()=>downloadJson("edge-network-simulation.json",data));
    target.append(bars,download);
    $("#sim-playback").hidden=false;
    simSelect();
    status($("#sim-status"),"Simulation complete. Move the virtual clock or play the trace.");
  } catch(error) { status($("#sim-status"),error.message,true); }
  finally { button.disabled=false; }
});
let edgeBusy = false;
let edgeConfigLoaded = false;
function renderEdge(result, syncConfig = false) {
  if (syncConfig || !edgeConfigLoaded) {
    $("#edge-policy").value = result.policy;
    const symbols = $("#edge-symbols");
    symbols.replaceChildren();
    for (const ticker of result.supported) {
      const label = element("label", undefined, "edge-option");
      const check = element("input");
      check.type = "checkbox"; check.value = ticker; check.checked = result.watchlist.includes(ticker);
      label.append(check, document.createTextNode(ticker)); symbols.append(label);
    }
    edgeConfigLoaded = true;
  }
  const metrics = $("#edge-metrics");
  metrics.replaceChildren();
  for (const [name, value] of [["Pending", result.pending], ["Failed retained", result.failed], ["Capacity", result.capacity]]) {
    const card = element("article");
    card.append(element("p", name, "eyebrow"), element("strong", String(value)));
    metrics.append(card);
  }
  const queue = $("#edge-queue");
  queue.replaceChildren(element("h3", "Next jobs"));
  if (result.queue.length) queue.append(table("Current queue order", ["Issuer", "Published", "Priority", "Wait"], result.queue.map(r =>
    [r.ticker, r.published_at, String(r.priority), r.age_seconds + " s"])));
  else queue.append(element("p", "No pending received announcements.", "small muted"));
  const history = $("#edge-history");
  history.replaceChildren(element("h3", "Recent actual processing"));
  if (result.recent.length) {
    history.append(table("Local processing history", ["Issuer", "Queue wait", "Processing", "Revision"], result.recent.map(r =>
      [r.ticker, r.wait_ms + " ms", r.processing_ms + " ms", r.delta.state.replaceAll("_", " ")])));
    for (const r of result.recent.filter(r => r.delta.state === "linked_excerpt_changed")) {
      const delta = r.delta;
      const section = element("details");
      section.append(element("summary", r.ticker + " · Same-link excerpt changed"), element("p", delta.note, "small muted"));
      section.append(element("p", "Removed figures: " + (delta.removed_figures.join(", ") || "None detected")),
        element("p", "Added figures: " + (delta.added_figures.join(", ") || "None detected")));
      for (const change of delta.changes) section.append(element("p", "Before: " + change.before), element("p", "After: " + change.after));
      if (delta.model_comparison === "comparable") {
        const labels = values => values.map(e => e.category + " / " + e.stage).join(", ") || "None";
        section.append(element("p", "Model event labels removed: " + labels(delta.removed_event_labels)),
          element("p", "Model event labels added: " + labels(delta.added_event_labels)),
          element("p", "These are inferred labels, not verified economic effects.", "small muted"));
      } else if (delta.model_comparison) section.append(element("p", "Model comparison withheld: differing model version or source scope.", "small muted"));
      history.append(section);
    }
  } else history.append(element("p", "No live jobs processed yet.", "small muted"));
}
async function loadEdge(syncConfig = false) {
  try { renderEdge(await api("/api/edge/status"), syncConfig); }
  catch (error) { status($("#edge-status"), error.message, true); }
}
async function edgeAction(action, payload) {
  if (edgeBusy) return;
  edgeBusy = true;
  const controls = [...$("#sentinel-panel").querySelectorAll("button")];
  controls.forEach(button => button.disabled = true);
  status($("#edge-status"), "Running local " + action + "…");
  try {
    const result = await api("/api/edge/" + action, {method:"POST",
      ...(payload ? {headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)} : {})});
    if (action === "benchmark") {
      const target = $("#edge-results");
      const rows = Object.entries(result.runs).flatMap(([policy,runs]) => runs.map((r,i) =>
        [policy + " " + (i+1), r.first_relevant_ms + " ms", r.all_relevant_ms + " ms", r.total_ms + " ms"]));
      target.replaceChildren(element("h3", "Measured scheduling experiment"), element("p", result.scope, "small muted"),
        table("Actual host model runs", ["Policy / run", "First relevant", "All relevant", "Total"], rows),
        element("p", "Equivalent category outputs: " + (result.same_categories ? "Yes" : "No")));
      const download = element("button", "Download benchmark evidence", "button");
      download.type = "button"; download.addEventListener("click", () => downloadJson("edge-scheduling.json", result));
      target.append(download);
    } else if (action === "drain") {
      if (result.state === "busy") { status($("#edge-status"), "Processor busy; retry."); return; }
      await refresh();
    }
    await loadEdge(action === "config");
    status($("#edge-status"), action === "drain" ? "Processed " + result.imported + " · pending " + result.pending +
      " · batch budget " + result.budget.max_jobs + " · CPU " + result.budget.cpu_percent + "%" : "Complete.");
  } catch (error) { status($("#edge-status"), error.message, true); }
  finally { edgeBusy = false; controls.forEach(button => button.disabled = false); }
}
$("#edge-refresh").addEventListener("click", () => loadEdge());
$("#edge-config").addEventListener("submit", event => {
  event.preventDefault();
  edgeAction("config", {watchlist: [...$("#edge-symbols").querySelectorAll("input:checked")].map(i => i.value), policy: $("#edge-policy").value});
});
$("#edge-drain").addEventListener("click", () => edgeAction("drain"));
$("#edge-benchmark").addEventListener("click", () => edgeAction("benchmark"));

function renderState() {
  $("#live-feed-panel").hidden = !state.live_feed_available;
  $("#sentinel-panel").hidden = !state.live_feed_available;
  $("#cooperative-panel").hidden = !state.live_feed_available;
  if (state.live_feed_available && !edgeConfigLoaded && !$("#view-lab").hidden) loadEdge();
  const hosted = state.mode === "hosted";
  $("#workspace-scope").textContent = hosted ? "Private browser workspace" : "Local workspace";
  $("#storage-note").hidden = !hosted || state.storage !== "sqlite";
  $("#clear-workspace").hidden = !hosted;
  $("#clear-workspace").disabled = workspaceBusy || replayBusy || analysisBusy;
  $("#edge-scope").textContent = hosted
    ? "While the source link is disconnected, the website can still analyze documents on the server."
    : "While the source link is disconnected, the workspace can still analyze documents on this computer.";
  const stats = {count: state.disclosures.length, link: state.connected ? "Connected" : "Disconnected", pending: state.pending, duplicates: state.duplicates};
  for (const item of document.querySelectorAll("[data-stat]")) item.textContent = stats[item.dataset.stat];
  for (const item of document.querySelectorAll('[data-stat="link"]')) item.dataset.online = String(state.connected);
  const progress = $("#replay-progress");
  progress.max = state.total_packets;
  progress.value = state.cursor;
  $("#replay-count").textContent = state.cursor + " of " + state.total_packets + " packets advanced";
  $("#toggle-link").textContent = state.connected ? "Disconnect link" : "Reconnect link";
  $("#toggle-link").disabled = replayBusy || workspaceBusy;
  $("#step-replay").disabled = replayBusy || workspaceBusy || state.cursor >= state.total_packets;
  $("#reset-replay").disabled = replayBusy || workspaceBusy || (!state.cursor && state.connected);
  $("#analysis-form [type=submit]").disabled = analysisBusy || workspaceBusy;
  $("#virtual-display").src = "/api/display.png?v=" + displayRevision;
  $("#virtual-display").alt = "Virtual status display. Source link " + stats.link.toLowerCase() + ", " + state.pending + " pending packets.";
  $("#virtual-display").hidden = false;
  renderInbox();
}

async function refresh() {
  const previous = state;
  state = await api("/api/inbox");
  displayRevision += 1;
  if (!previous) {
    addOptions($("#filter-ticker"), state.tickers);
    addOptions($("#filter-category"), state.labels);
    addOptions($("#form-ticker"), state.tickers);
    restoreInboxView();
  }
  renderState();
}

async function replay(action, payload) {
  if (!state || replayBusy) return;
  replayBusy = true;
  renderState();
  const note = $("#replay-status");
  try {
    const result = await api("/api/replay/" + action, {
      method: "POST",
      ...(payload ? {headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)} : {}),
    });
    await refresh();
    if (action === "reset") status(note, "Replay reset. Imported disclosures are still in the inbox.");
    else if (!result.connected) status(note, `${result.pending} pending. Document analysis is available.`);
    else status(note, `${result.pending} pending. ${result.duplicates} duplicate packets suppressed.`);
  } catch (error) {
    status(note, error.message, true);
  } finally {
    replayBusy = false;
    renderState();
  }
}

function table(caption, headings, rows, rowHeaders = false) {
  const output = element("table");
  output.append(element("caption", caption));
  const head = element("thead");
  const header = element("tr");
  for (const heading of headings) {
    const cell = element("th", heading);
    cell.scope = "col";
    header.append(cell);
  }
  head.append(header);
  const body = element("tbody");
  for (const values of rows) {
    const row = element("tr");
    values.forEach((value, index) => {
      const cell = element(rowHeaders && index === 0 ? "th" : "td");
      if (rowHeaders && index === 0) cell.scope = "row";
      if (value instanceof HTMLElement) cell.append(value);
      else cell.textContent = value;
      row.append(cell);
    });
    body.append(row);
  }
  output.append(head, body);
  const wrap = element("div", undefined, "table-wrap");
  wrap.tabIndex = 0;
  wrap.setAttribute("aria-label", caption);
  wrap.append(output);
  return wrap;
}

function evaluation(report) {
  const host = report.host;
  const data = report.dataset;
  const layout = $("#evaluation-results");
  layout.replaceChildren();
  const context = element("div", undefined, "actions report-context");
  const download = element("button", "Download report", "button");
  download.type = "button";
  download.addEventListener("click", () => downloadJson("stockbrief-benchmark.json", report));
  context.append(element("p", data.train + " training · " + data.calibration + " calibration · " + data.test + " held out. " + data.scope + ".", "small muted"), download);
  layout.append(context);
  const scores = element("div", undefined, "model-scores");
  for (const result of report.reports) {
    const card = element("div", undefined, "model-score");
    const bar = element("meter");
    bar.min = 0;
    bar.max = 1;
    bar.value = result.macro_f1;
    bar.setAttribute("aria-label", result.name + " macro F1");
    card.append(element("strong", result.macro_f1.toFixed(3)), element("span", result.name), bar);
    scores.append(card);
  }
  layout.append(element("h3", "Macro F1 on the held-out set"), scores);
  layout.append(table("Held-out results", ["Model", "Macro F1", "Coverage", "Accuracy when covered", "Review recall", "p50 / p95"], report.reports.map((r) => [
    r.name, r.macro_f1.toFixed(3), percent(r.coverage), percent(r.selective_accuracy), percent(r.review_recall), r.latency_ms.p50 + " / " + r.latency_ms.p95 + " ms",
  ]), true));

  for (const checks of report.reports) {
    const section = element("details");
    section.append(element("summary", checks.name + " · class details"));
    section.append(table(
      "Confusion matrix · rows actual, columns predicted",
      ["Actual", ...checks.classes.map((r) => report.labels[r.label])],
      checks.matrix.map((row, index) => [report.labels[checks.classes[index].label], ...row]),
      true,
    ));
    section.append(table(
      "Per-class scores", ["Class", "Precision", "Recall", "F1", "Cases"],
      checks.classes.map((r) => [report.labels[r.label], r.precision, r.recall, r.f1, r.support]),
      true,
    ));
    if (checks.artifact) {
      const artifact = checks.artifact;
      section.append(element("p", artifact.repository + " · Revision " + artifact.revision.slice(0, 12) +
        " · ONNX Runtime " + artifact.onnxruntime_version + " · " + artifact.execution_providers.join(", "), "small muted"));
    }
    if (checks.historical_cases) {
      section.append(table(
        "Historical category check", ["Original source", "Expected category", "Observed category", "Matches expected"],
        checks.historical_cases.map((r) => [
          sourceLink(r.source, r.title), report.labels[r.expected_category], report.labels[r.observed_category],
          r.category_matched ? "Yes" : "No",
        ]),
        true,
      ));
    }
    layout.append(section);
  }
  layout.append(table("Historical case study", ["Original source", "Observed category", "Observed stage", "Matches expected"],
    report.historical_cases.map((r) => [sourceLink(r.source, r.title), report.labels[r.observed_category], r.observed_stage, r.matched ? "Yes" : "No"]), true));
  if (host?.available_logical_cpus != null && host?.process_rss_mib != null) {
    layout.append(element("p", host.platform + " · " + host.available_logical_cpus + " available logical CPUs · " + host.process_rss_mib + " MiB RSS. Host measurements.", "small muted"));
  }
}

const form = $("#analysis-form");

function setImportMethod(method) {
  const upload = method === "file";
  form.elements.method.value = method;
  $("#paste-field").hidden = upload;
  $("#upload-field").hidden = !upload;
  form.elements.text.disabled = upload;
  form.elements.text.required = !upload;
  form.elements.file.disabled = !upload;
  form.elements.file.required = upload;
}

async function loadWorkspace() {
  const retry = $("#retry-workspace");
  retry.disabled = true;
  try {
    await refresh();
    globalError(null);
  } catch (error) {
    globalError(error);
  } finally {
    retry.disabled = false;
  }
}

$(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  $("#main").focus();
});

addEventListener("hashchange", presentView);
$("#retry-workspace").addEventListener("click", loadWorkspace);
$("#filter-search").addEventListener("input", () => renderInbox());
$("#filter-ticker").addEventListener("change", () => renderInbox());
$("#filter-category").addEventListener("change", () => renderInbox());
$("#toggle-link").addEventListener("click", () => replay("link", {connected: !state.connected}));
$("#step-replay").addEventListener("click", () => replay("step"));
$("#reset-replay").addEventListener("click", () => replay("reset"));
$("#clear-workspace").addEventListener("click", async () => {
  if (!window.confirm("Clear imported disclosures and reset the replay for this browser workspace?")) return;
  workspaceBusy = true;
  renderState();
  const note = $("#workspace-status");
  try {
    await api("/api/workspace/clear", {method: "POST"});
    selection = undefined;
    inboxControls.forEach((id) => { $("#" + id).value = ""; });
    form.reset();
    setImportMethod("text");
    $("#analysis-result").replaceChildren(element("h2", "Analysis result"), element("p", "Add a disclosure to see a new result.", "muted"));
    status($("#analysis-status"), "");
    status($("#replay-status"), "");
    await refresh();
    status(note, "Browser workspace cleared.");
  } catch (error) {
    status(note, error.message, true);
  } finally {
    workspaceBusy = false;
    renderState();
  }
});
for (const option of form.elements.method) {
  option.addEventListener("change", () => setImportMethod(option.value));
}
$("#load-example").addEventListener("click", () => {
  const fields = form.elements;
  setImportMethod("text");
  fields.ticker.value = "OTHER";
  fields.title.value = "Example acquisition disclosure";
  fields.disclosure_date.value = "";
  fields.source.value = "";
  fields.url.value = "";
  fields.file.value = "";
  fields.text.value = "The company signed a definitive agreement to acquire an engineering business for INR 240 crore. The transaction remains subject to regulatory approvals.";
  fields.text.focus();
});
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const note = $("#analysis-status");
  if (!state || analysisBusy || workspaceBusy) return;
  analysisBusy = true;
  renderState();
  status(note, "Analyzing the disclosure…");
  try {
    const result = await api("/api/analyze", {method: "POST", body: new FormData(form)});
    const pane = $("#analysis-result");
    detail(result.entry, pane);
    status(note, result.duplicate ? "This disclosure is already saved." : "Saved to your inbox.");
    selection = result.entry.id;
    inboxControls.forEach((id) => { $("#" + id).value = ""; });
    await refresh();
    if (matchMedia("(max-width: 760px)").matches) pane.focus();
  } catch (error) {
    status(note, error.message, true);
  } finally {
    analysisBusy = false;
    renderState();
  }
});
$("#run-evaluation").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const note = $("#evaluation-status");
  button.disabled = true;
  status(note, "Running the labeled fixtures…");
  try {
    evaluation(await api("/api/evaluation"));
    status(note, "Benchmark complete.");
  } catch (error) {
    status(note, error.message, true);
  } finally {
    button.disabled = false;
  }
});
presentView();
loadWorkspace();
