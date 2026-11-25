const $ = (selector) => document.querySelector(selector);
const viewText = {
  inbox: ["Disclosure workspace", "Review company announcements with the source beside each brief."],
  analyze: ["Analyze a disclosure", "Classify a filing and find its supporting evidence on this device."],
  lab: ["Edge lab", "Test local inference while the simulated source link is interrupted."],
  evaluation: ["Evaluation", "Check the model’s event labels, review decisions, and host performance."],
};
const originNames = {
  historical: "Historical paraphrase",
  synthetic: "Synthetic scenario",
  text: "Text import",
  pdf: "PDF import",
};
let state;
let selection;
let replayBusy = false;
let displayRevision = 0;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function dateLabel(raw) {
  if (!raw) return "";
  return new Intl.DateTimeFormat("en-IN", {dateStyle: "medium", timeZone: "UTC"}).format(new Date(raw));
}

function percent(value) {
  return value == null ? "—" : (100 * value).toFixed(1) + "%";
}

function status(target, message, isError = false) {
  target.textContent = message;
  target.classList.toggle("error", isError);
}

function globalError(error) {
  const target = $("#global-error");
  target.hidden = !error;
  target.textContent = error?.message || "";
}

async function api(url, options) {
  const response = await fetch(url, options);
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
  return response.json();
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
  $("#analyze-shortcut").hidden = current !== "inbox";
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

function detail(entry, pane) {
  const brief = entry.brief;
  const heading = element("div", undefined, "chips");
  heading.append(element("span", entry.ticker, "chip"), element("span", originNames[entry.origin] || "Local import", "chip subtle"));
  if (brief.category === "review") heading.append(element("span", "Needs review", "chip review"));
  const meta = [entry.source, dateLabel(entry.date)].filter(Boolean).join(" · ");
  const facts = element("dl");
  facts.append(detailField("Event category", brief.category_name), detailField("Disclosure stage", brief.stage));
  pane.replaceChildren(heading, element("h2", entry.title), element("p", meta, "source-meta"), facts);
  if (brief.category === "review") {
    pane.append(element("p", "Leading candidate: " + brief.candidate_name + ". Read the excerpt before using the label.", "small muted"));
  }
  if (brief.chunk_limit_reached) {
    pane.append(element("p", "The analysis reached its 240-chunk limit. Later text may need a separate review.", "small muted"));
  }

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

  if (brief.figures.length) {
    const figures = element("div", undefined, "amounts");
    const chips = element("div", undefined, "chips");
    const figuresByValue = new Map(brief.figures.map((figure) => [figure.value, figure]));
    for (const figure of figuresByValue.values()) {
      const locator = entry.origin === "historical" ? "Excerpt " : "Page ";
      chips.append(element("span", figure.value + " · " + locator + figure.page, "chip subtle"));
    }
    figures.append(element("h3", "Figures in the source text", "eyebrow"), chips);
    pane.append(figures);
  }

  if (entry.url) pane.append(sourceLink(entry.url, "Open original source ↗"));
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
      status(result, "Running the local transformer…");
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
  const entries = state.disclosures.filter((d) => (!ticker || d.ticker === ticker) && (!category || d.brief.category === category));
  $("#filter-count").textContent = entries.length + " shown";
  if (!entries.some((entry) => entry.id === selection)) selection = entries[0]?.id;
  const rows = $("#disclosure-list");
  const pane = $("#disclosure-detail");
  rows.replaceChildren();
  for (const entry of entries) {
    const row = element("button", undefined, "document-row");
    row.type = "button";
    row.dataset.id = entry.id;
    row.setAttribute("aria-pressed", String(entry.id === selection));
    const issuer = [entry.ticker, dateLabel(entry.date)].filter(Boolean).join(" · ");
    const event = entry.brief.category_name + (entry.brief.stage === "Unclear" ? "" : " · " + entry.brief.stage);
    row.append(element("span", issuer), element("strong", entry.title), element("span", event));
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
  if (selected) detail(selected, pane);
  else {
    rows.append(element("li", "No disclosures match these filters.", "muted"));
    pane.replaceChildren(element("h2", "No disclosure selected"), element("p", "Adjust the inbox filters to find a disclosure.", "muted"));
  }
}

function renderState() {
  const stats = {count: state.disclosures.length, link: state.connected ? "Connected" : "Disconnected", pending: state.pending, duplicates: state.duplicates};
  for (const item of document.querySelectorAll("[data-stat]")) item.textContent = stats[item.dataset.stat];
  const progress = $("#replay-progress");
  progress.max = state.total_packets;
  progress.value = state.cursor;
  $("#replay-count").textContent = state.cursor + " of " + state.total_packets + " packets advanced";
  $("#toggle-link").textContent = state.connected ? "Disconnect link" : "Reconnect link";
  $("#toggle-link").disabled = replayBusy;
  $("#step-replay").disabled = replayBusy || state.cursor >= state.total_packets;
  $("#reset-replay").disabled = replayBusy || (!state.cursor && state.connected);
  $("#virtual-display").src = "/api/display.png?v=" + (++displayRevision);
  $("#virtual-display").alt = "Virtual status display. Source link " + stats.link.toLowerCase() + ", " + state.pending + " pending packets.";
  renderInbox();
}

async function refresh() {
  const previous = state;
  state = await api("/api/inbox");
  if (!previous) {
    addOptions($("#filter-ticker"), state.tickers);
    addOptions($("#filter-category"), state.labels);
    addOptions($("#form-ticker"), state.tickers);
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
    if (action === "reset") status(note, "Replay reset. Local imports are still in the inbox.");
    else if (!result.connected) status(note, "Source link disconnected. Local analysis is available.");
    else status(note, "Source link connected. " + result.pending + " pending packets.");
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
  wrap.append(output);
  return wrap;
}

function evaluation(report) {
  const host = report.host;
  const data = report.dataset;
  const layout = $("#evaluation-results");
  layout.replaceChildren();
  layout.append(element("p", data.train + " training · " + data.calibration + " calibration · " + data.test + " held out. " + data.scope + ".", "small muted"));
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
    layout.append(section);
  }
  layout.append(table("Historical case study", ["Original source", "Observed category", "Observed stage", "Matches expected"],
    report.historical_cases.map((r) => [sourceLink(r.source, r.title), report.labels[r.observed_category], r.observed_stage, r.matched ? "Yes" : "No"]), true));
  layout.append(element("p", host.platform + " · " + host.available_logical_cpus + " available logical CPUs · " + host.process_rss_mib + " MiB RSS. Host measurements.", "small muted"));
}

$(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  $("#main").focus();
});

addEventListener("hashchange", presentView);
$("#filter-ticker").addEventListener("change", () => renderInbox());
$("#filter-category").addEventListener("change", () => renderInbox());
$("#toggle-link").addEventListener("click", () => replay("link", {connected: !state.connected}));
$("#step-replay").addEventListener("click", () => replay("step"));
$("#reset-replay").addEventListener("click", () => replay("reset"));
$("#load-example").addEventListener("click", () => {
  const fields = $("#analysis-form").elements;
  fields.ticker.value = "OTHER";
  fields.title.value = "Example acquisition disclosure";
  fields.disclosure_date.value = "";
  fields.source.value = "";
  fields.url.value = "";
  fields.file.value = "";
  fields.text.value = "The company signed a definitive agreement to acquire an engineering business for INR 240 crore. The transaction remains subject to regulatory approvals.";
  fields.text.focus();
});
$("#analysis-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("[type=submit]");
  const note = $("#analysis-status");
  button.disabled = true;
  status(note, "Analyzing on this device…");
  try {
    const result = await api("/api/analyze", {method: "POST", body: new FormData(form)});
    detail(result.entry, $("#analysis-result"));
    status(note, result.duplicate ? "This disclosure is already saved." : "Saved to the local inbox.");
    selection = result.entry.id;
    await refresh();
  } catch (error) {
    status(note, error.message, true);
  } finally {
    button.disabled = false;
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
refresh().then(() => globalError(null)).catch(globalError);
