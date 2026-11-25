const $ = (selector) => document.querySelector(selector);
const viewText = {
  inbox: ["Disclosure workspace", "Find the event, its stage, and the source passage behind each company announcement."],
  analyze: ["Analyze a disclosure", "Classify a company filing and inspect the passages used in the result."],
  lab: ["Edge lab", "Queue, resend, and deduplicate disclosures on a simulated source link."],
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
    if (link.dataset.view === current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  $("#view-title").textContent = viewText[current][0];
  $("#view-description").textContent = viewText[current][1];
  $(".page-head").dataset.current = current;
  $("#hero-actions").hidden = current !== "inbox";
  $("#hero-process").hidden = current !== "inbox";
  document.title = viewText[current][0] + " · StockBrief Edge";
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
  panel.append(element("h3", "Observed market context"));
  const note = element("p", "Checking completed trading sessions…", "small muted");
  note.setAttribute("role", "status");
  panel.append(note);
  const signed = (n, unit) => n == null ? "Unavailable" : (n > 0 ? "+" : "") + n.toFixed(2) + unit;
  api("/api/market-context/" + encodeURIComponent(entry.ticker) + "?filing_date=" + encodeURIComponent(entry.date))
    .then(result => {
      const metrics = element("dl");
      metrics.append(detailField("Company return", signed(result.symbol_return_pct, "%")),
        detailField("Nifty 50 return", signed(result.benchmark_return_pct, "%")),
        detailField("Relative return", signed(result.relative_return_pp, " pp")));
      note.textContent = result.before && result.after
        ? dateLabel(result.before.date) + " to " + dateLabel(result.after.date) + " · " + result.state
        : result.state + " · " + (result.reason || "No matching sessions").replaceAll("_", " ");
      panel.append(metrics);
      if (result.before && result.after) {
        panel.append(element("p", "Company closes: ₹" + result.before.close + " → ₹" + result.after.close, "small"));
        const chart = $("#market-symbol");
        if (chart && !$("#market-panel").hidden) {
          const button = element("button", "Show issuer price chart", "button");
          button.type = "button";
          button.addEventListener("click", () => {
            chart.value = entry.ticker;
            chart.dispatchEvent(new Event("change"));
            $("#market-panel").scrollIntoView({block: "start"});
          });
          panel.append(button);
        }
      }
      const list = element("ul");
      result.limitations.forEach(item => list.append(element("li", item)));
      panel.append(element("p", "Upstox · source checked " + (result.symbol_checked_at || "unavailable"), "small muted"), list);
    }).catch(error => status(note, error.message, true));
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

  if (entry.origin.startsWith("nse-") && entry.date) pane.append(marketContextPanel(entry));

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
async function pollLive() {
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
      result.pending + " pending in recent window · " + result.errors + " processing warnings";
    status(note, (result.state === "unavailable" ? "Feed unavailable; saved briefs retained." : summary) +
      " · Last successful check: " + (result.last_success || "none") +
      (result.throttled ? " · Cached refresh status" : ""), result.state === "unavailable");
    if (result.imported || result.throttled) await refresh();
  } catch (error) {
    status(note, error.message, true);
  } finally {
    liveBusy = false;
    button.disabled = false;
  }
}
$("#live-refresh").addEventListener("click", pollLive);
$("#live-toggle").addEventListener("click", () => {
  liveEnabled = !liveEnabled;
  $("#live-toggle").textContent = liveEnabled ? "Stop auto refresh" : "Start auto refresh";
  if (liveEnabled) pollLive();
});
setInterval(() => {
  if (liveEnabled && !document.hidden && !$("#view-inbox").hidden && Date.now() - liveStarted >= 120000) pollLive();
}, 5000);

function renderState() {
  $("#live-feed-panel").hidden = !state.live_feed_available;
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
